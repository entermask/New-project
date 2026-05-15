#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run an OmniVoice server benchmark matrix and write a report.

Required unless present in .env:
  API_TOKEN                    Bearer token for the API

Common overrides:
  BASE_URL                     Default: http://127.0.0.1:8001
  REF_AUDIO_URL                Default: German sample URL from Hailuo CDN
  REF_TEXT                     Optional reference transcript
  TEXT                         Text to synthesize
  LANGUAGE                     Default: de
  REQUESTS                     Default: 20
  PORT                         Default: 8001
  APP_DIR                      Default: repo root
  VENV_DIR                     Default: $HOME/venvs/omnivoice-api
  REPORT_ROOT                  Default: APP_DIR/reports
  HEALTH_TIMEOUT_SECONDS       Default: 900

Matrix overrides, space-separated:
  MATRIX_ACCELERATIONS         Default: base triton
  MATRIX_CONCURRENCIES         Default: 4 6 8 12
  MATRIX_STEPS                 Default: 16 32
  MATRIX_SPEEDS                Default: 1.0 1.1

Example smoke test:
  REQUESTS=2 MATRIX_CONCURRENCIES=4 MATRIX_STEPS=16 MATRIX_SPEEDS=1.1 MATRIX_ACCELERATIONS=base ./scripts/run_server_benchmark_report.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_APP_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

APP_DIR="${APP_DIR:-$DEFAULT_APP_DIR}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/omnivoice-api}"
PORT="${PORT:-8001}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT}"
REF_AUDIO_URL="${REF_AUDIO_URL:-https://cdn.hailuoai.video/moss/prod/2026-05-14-05/moss-audio/voice/u_2054676293873570458/demo/1778708587932629888-397944920203503_German.mp3}"
REF_TEXT="${REF_TEXT:-}"
TEXT="${TEXT:-Guten Tag, dies ist ein OmniVoice Benchmark auf dem Server.}"
LANGUAGE="${LANGUAGE:-de}"
REQUESTS="${REQUESTS:-20}"
FORMAT="${FORMAT:-mp3}"
MODE="${MODE:-poll}"
POLL_INTERVAL="${POLL_INTERVAL:-0.5}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
REPORT_ROOT="${REPORT_ROOT:-$APP_DIR/reports}"
RESTORE_SERVICE="${RESTORE_SERVICE:-1}"

MATRIX_ACCELERATIONS="${MATRIX_ACCELERATIONS:-base triton}"
MATRIX_CONCURRENCIES="${MATRIX_CONCURRENCIES:-4 6 8 12}"
MATRIX_STEPS="${MATRIX_STEPS:-16 32}"
MATRIX_SPEEDS="${MATRIX_SPEEDS:-1.0 1.1}"

ENV_FILE="$APP_DIR/.env"
RUN_TMUX="$APP_DIR/scripts/run_tmux.sh"
BENCHMARK="$APP_DIR/scripts/benchmark_tts.py"

if [[ ! -x "$RUN_TMUX" ]]; then
  echo "Missing executable $RUN_TMUX" >&2
  exit 1
fi

if [[ ! -f "$BENCHMARK" ]]; then
  echo "Missing $BENCHMARK" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "python3 is required" >&2
    exit 1
  fi
fi

if [[ -z "${API_TOKEN:-}" && -f "$ENV_FILE" ]]; then
  API_TOKEN="$(
    "$PYTHON_BIN" - "$ENV_FILE" <<'PY'
import shlex
import sys
from pathlib import Path

for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    if key.strip() == "API_TOKEN":
        print(shlex.split("x=" + value, posix=True)[0].split("=", 1)[1])
        break
PY
  )"
fi

if [[ -z "${API_TOKEN:-}" ]]; then
  echo "API_TOKEN must be set in the environment or $ENV_FILE" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="$REPORT_ROOT/tts-benchmark-$TIMESTAMP"
CASE_DIR="$REPORT_DIR/cases"
HEALTH_DIR="$REPORT_DIR/health"
AUDIO_DIR="$REPORT_DIR/audio"
LOG_DIR="$REPORT_DIR/logs"
REPORT_MD="$REPORT_DIR/report.md"
ENV_BACKUP="$REPORT_DIR/env.original"
HAD_ENV_FILE=0
FAILED_CASES=0

mkdir -p "$CASE_DIR" "$HEALTH_DIR" "$AUDIO_DIR" "$LOG_DIR"

if [[ -f "$ENV_FILE" ]]; then
  cp "$ENV_FILE" "$ENV_BACKUP"
  HAD_ENV_FILE=1
else
  : > "$ENV_BACKUP"
fi

restore_env() {
  set +e
  if [[ "$HAD_ENV_FILE" == "1" ]]; then
    cp "$ENV_BACKUP" "$ENV_FILE"
  else
    rm -f "$ENV_FILE"
  fi

  if [[ "$RESTORE_SERVICE" == "1" && "$HAD_ENV_FILE" == "1" ]]; then
    RESTART=1 APP_DIR="$APP_DIR" VENV_DIR="$VENV_DIR" PORT="$PORT" "$RUN_TMUX" >/dev/null 2>&1
  fi
}
trap restore_env EXIT

write_case_env() {
  local acceleration="$1"
  local concurrency="$2"

  grep -Ev '^(OMNIVOICE_ACCELERATION|OMNIVOICE_CONCURRENCY|OMNIVOICE_BATCH_SIZE|OMNIVOICE_ENABLE_BATCHING|OMNIVOICE_SKIP_MODEL_LOAD|API_TOKEN)=' "$ENV_BACKUP" > "$ENV_FILE" || true
  {
    printf 'API_TOKEN=%q\n' "$API_TOKEN"
    printf 'OMNIVOICE_ACCELERATION=%q\n' "$acceleration"
    printf 'OMNIVOICE_CONCURRENCY=%q\n' "$concurrency"
    printf 'OMNIVOICE_BATCH_SIZE=%q\n' "$concurrency"
    printf 'OMNIVOICE_ENABLE_BATCHING=1\n'
    printf 'OMNIVOICE_SKIP_MODEL_LOAD=0\n'
  } >> "$ENV_FILE"
}

wait_for_health() {
  local expected_acceleration="$1"
  local expected_concurrency="$2"
  local expected_batch="$3"
  local output_path="$4"
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
  local tmp_path="$output_path.tmp"

  while (( SECONDS < deadline )); do
    if curl -fsS "$BASE_URL/health" -o "$tmp_path" >/dev/null 2>&1; then
      if "$PYTHON_BIN" - "$tmp_path" "$expected_acceleration" "$expected_concurrency" "$expected_batch" <<'PY'
import json
import sys

path, expected_acceleration, expected_concurrency, expected_batch = sys.argv[1:5]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)

ok = (
    data.get("status") == "ok"
    and data.get("model_loaded") is True
    and data.get("acceleration") == expected_acceleration
    and str(data.get("max_concurrency")) == expected_concurrency
    and str(data.get("batch_size")) == expected_batch
)
raise SystemExit(0 if ok else 1)
PY
      then
        mv "$tmp_path" "$output_path"
        return 0
      fi
    fi
    sleep 5
  done

  if [[ -f "$tmp_path" ]]; then
    mv "$tmp_path" "$output_path"
  fi
  return 1
}

append_case_row() {
  local case_id="$1"
  local acceleration="$2"
  local concurrency="$3"
  local step="$4"
  local speed="$5"
  local result_json="$6"
  local status="$7"

  "$PYTHON_BIN" - "$case_id" "$acceleration" "$concurrency" "$step" "$speed" "$result_json" "$status" <<'PY' >> "$REPORT_MD"
import json
import sys
from pathlib import Path

case_id, acceleration, concurrency, step, speed, result_json, status = sys.argv[1:8]
summary = {}
path = Path(result_json)
if path.exists():
    try:
        summary = json.loads(path.read_text(encoding="utf-8")).get("summary", {})
    except Exception:
        summary = {}

def value(name):
    return summary.get(name, "")

print(
    "| {case} | {accel} | {conc} | {batch} | {step} | {speed} | {requests} | {ok} | {failed} | {rps} | {avg} | {p50} | {p90} | {p95} | {p99} | {max_ms} | {hits} | {misses} | {status} |".format(
        case=case_id,
        accel=acceleration,
        conc=concurrency,
        batch=concurrency,
        step=step,
        speed=speed,
        requests=value("requests"),
        ok=value("ok"),
        failed=value("failed"),
        rps=value("requests_per_s"),
        avg=value("avg_ms"),
        p50=value("p50_ms"),
        p90=value("p90_ms"),
        p95=value("p95_ms"),
        p99=value("p99_ms"),
        max_ms=value("max_ms"),
        hits=value("cache_hits"),
        misses=value("cache_misses"),
        status=status,
    )
)
PY
}

{
  echo "# OmniVoice TTS Benchmark Report"
  echo
  echo "- Started: $(date -Iseconds)"
  echo "- Base URL: $BASE_URL"
  echo "- Reference audio: $REF_AUDIO_URL"
  echo "- Text: $TEXT"
  echo "- Language: $LANGUAGE"
  echo "- Requests per case: $REQUESTS"
  echo "- Format: $FORMAT"
  echo "- Mode: $MODE"
  echo "- Hybrid: skipped"
  echo
  echo "## Results"
  echo
  echo "| Case | Accel | Concurrency | Batch | Step | Speed | Requests | OK | Failed | RPS | Avg ms | P50 | P90 | P95 | P99 | Max ms | Cache hit | Cache miss | Status |"
  echo "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
} > "$REPORT_MD"

read -r -a ACCELERATIONS <<< "$MATRIX_ACCELERATIONS"
read -r -a CONCURRENCIES <<< "$MATRIX_CONCURRENCIES"
read -r -a STEPS <<< "$MATRIX_STEPS"
read -r -a SPEEDS <<< "$MATRIX_SPEEDS"

echo "Writing report to $REPORT_DIR"

for acceleration in "${ACCELERATIONS[@]}"; do
  if [[ "$acceleration" == "hybrid" ]]; then
    echo "Skipping hybrid acceleration"
    continue
  fi

  for concurrency in "${CONCURRENCIES[@]}"; do
    write_case_env "$acceleration" "$concurrency"
    echo "Restarting API: acceleration=$acceleration concurrency=$concurrency batch=$concurrency"
    restart_log="$LOG_DIR/restart-${acceleration}-c${concurrency}.log"
    set +e
    RESTART=1 APP_DIR="$APP_DIR" VENV_DIR="$VENV_DIR" PORT="$PORT" "$RUN_TMUX" > "$restart_log" 2>&1
    restart_rc=$?
    set -e

    if (( restart_rc != 0 )); then
      echo "Restart failed for acceleration=$acceleration concurrency=$concurrency, see $restart_log"
      for step in "${STEPS[@]}"; do
        for speed in "${SPEEDS[@]}"; do
          speed_id="${speed//./p}"
          case_id="${acceleration}-c${concurrency}-step${step}-speed${speed_id}"
          append_case_row "$case_id" "$acceleration" "$concurrency" "$step" "$speed" "$CASE_DIR/$case_id.json" "restart_failed_rc_${restart_rc}"
          FAILED_CASES=$((FAILED_CASES + 1))
        done
      done
      continue
    fi

    for step in "${STEPS[@]}"; do
      for speed in "${SPEEDS[@]}"; do
        speed_id="${speed//./p}"
        case_id="${acceleration}-c${concurrency}-step${step}-speed${speed_id}"
        result_json="$CASE_DIR/$case_id.json"
        result_csv="$CASE_DIR/$case_id.csv"
        case_audio_dir="$AUDIO_DIR/$case_id"
        before_health="$HEALTH_DIR/$case_id-before.json"
        after_health="$HEALTH_DIR/$case_id-after.json"
        case_log="$LOG_DIR/$case_id.log"
        status="ok"

        echo "Running $case_id"
        if ! wait_for_health "$acceleration" "$concurrency" "$concurrency" "$before_health"; then
          echo "Health check timed out for $case_id" | tee "$case_log"
          status="health_timeout"
          FAILED_CASES=$((FAILED_CASES + 1))
          append_case_row "$case_id" "$acceleration" "$concurrency" "$step" "$speed" "$result_json" "$status"
          continue
        fi

        mkdir -p "$case_audio_dir"
        benchmark_args=(
          "$PYTHON_BIN" "$BENCHMARK"
          --base-url "$BASE_URL"
          --token "$API_TOKEN"
          --ref-audio-url "$REF_AUDIO_URL"
          --text "$TEXT"
          --requests "$REQUESTS"
          --concurrency "$concurrency"
          --num-step "$step"
          --speed "$speed"
          --format "$FORMAT"
          --mode "$MODE"
          --poll-interval "$POLL_INTERVAL"
          --timeout "$REQUEST_TIMEOUT"
          --output-dir "$case_audio_dir"
          --results-json "$result_json"
          --results-csv "$result_csv"
          --fail-on-error
        )

        if [[ -n "$LANGUAGE" ]]; then
          benchmark_args+=(--language "$LANGUAGE")
        fi
        if [[ -n "$REF_TEXT" ]]; then
          benchmark_args+=(--ref-text "$REF_TEXT")
        fi

        set +e
        "${benchmark_args[@]}" > "$case_log" 2>&1
        rc=$?
        set -e
        if (( rc != 0 )); then
          status="benchmark_failed_rc_${rc}"
          FAILED_CASES=$((FAILED_CASES + 1))
        fi

        curl -fsS "$BASE_URL/health" -o "$after_health" >/dev/null 2>&1 || true
        append_case_row "$case_id" "$acceleration" "$concurrency" "$step" "$speed" "$result_json" "$status"
      done
    done
  done
done

{
  echo
  echo "## Artifacts"
  echo
  echo "- Raw case results: \`$CASE_DIR\`"
  echo "- Health snapshots: \`$HEALTH_DIR\`"
  echo "- Audio samples: \`$AUDIO_DIR\`"
  echo "- Logs: \`$LOG_DIR\`"
  echo
  echo "Finished: $(date -Iseconds)"
  echo "Failed cases: $FAILED_CASES"
} >> "$REPORT_MD"

echo "Report written: $REPORT_MD"

if (( FAILED_CASES > 0 )); then
  exit 1
fi
