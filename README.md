# OmniVoice API

Standalone FastAPI wrappers for [OmniVoice](https://github.com/k2-fsa/OmniVoice) TTS and Qwen3-ASR STT. The repo now ships two independent servers intended to run on separate RTX 5090 machines; both default to port `6006` because they are deployed separately.

## Server Entrypoints

TTS:

```bash
cp .env.tts.example .env
./scripts/install_tts.sh
# or after install:
./scripts/run_tts.sh
```

On GPUHub hosts, keep heavy model cache on the system disk with `HF_HOME=$HOME/.cache/huggingface`, and keep generated reference-audio cache, job audio, and temp files on the data disk with `OMNIVOICE_CACHE_DIR` and `TMPDIR` under `$HOME/autodl-tmp`.

STT:

```bash
cp .env.stt.example .env
./scripts/install_stt.sh
# or after install:
./scripts/run_stt.sh
```

Legacy `uvicorn app:app` remains a TTS compatibility shim. Use `uvicorn omnivoice_api.stt.server:app --host 0.0.0.0 --port 6006` for STT.

## Qwen3-ASR STT

The STT server keeps the existing `/v1/stt/transcribe` and `/v1/stt/status/{job_id}` contract, but adds an optional multipart form field `model` with values `0.6b` or `1.7b`. If omitted, `QWEN_ASR_DEFAULT_MODEL=0.6b` is used. Startup loads both `Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B` unless `QWEN_ASR_SKIP_MODEL_LOAD=1`.

On the benchmarked RTX 5090 host with driver `570.169`, stable `qwen-asr[vllm]` works when vLLM is forced away from FlashAttention: `QWEN_ASR_VLLM_ATTENTION_BACKEND=TRITON_ATTN` and `QWEN_ASR_VLLM_MM_ENCODER_ATTN_BACKEND=TORCH_SDPA`. Because startup loads both 0.6B and 1.7B, the default `QWEN_ASR_GPU_MEMORY_UTILIZATION` is `0.35` per vLLM engine. The measured sweet spot for 30s audio is `QWEN_ASR_MAX_INFERENCE_BATCH_SIZE=32`; batch 64/96/128 runs without OOM in single-model direct benchmarks, but p95 latency jumps and throughput is less stable. The STT worker batches queued requests per model and reduces the dynamic batch size on CUDA OOM before retrying.

Run the RTX 5090 benchmark after the STT server is up:

```bash
python scripts/benchmark_stt_qwen.py \
  --base-url http://127.0.0.1:6006 \
  --token "$API_TOKEN" \
  --audio short.wav medium.wav long.wav
```

Benchmark JSON and Markdown reports are written to `benchmarks/stt-qwen3-5090/`.

## API

### `GET /health`

Returns model, GPU, chunk scheduler, and cache status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "gpu": "NVIDIA A100 80GB",
  "gpu_profile": "a100",
  "requested_gpu_profile": "auto",
  "dtype": "fp16",
  "cache_audio_count": 12,
  "cache_transcript_count": 12,
  "active_requests": 0,
  "active_generations": 12,
  "active_generation_batches": 1,
  "queued_generations": 12,
  "queued_chunks": 12,
  "running_chunks": 12,
  "outstanding_chunks": 24,
  "deferred_chunks": 0,
  "active_tts_jobs": 2,
  "chunk_size_chars": 200,
  "batch_size": 12,
  "batch_max_wait_ms": 50.0,
  "busy_backlog_chunks": 24,
  "tts_jobs": {
    "queued": 2,
    "running": 1,
    "succeeded": 20,
    "failed": 0
  },
  "acceleration": "triton"
}
```

### `POST /v1/tts`

Auth:

```http
Authorization: Bearer <API_TOKEN>
```

Request:

```json
{
  "chunks": ["Đoạn thứ nhất.", "Đoạn thứ hai."],
  "ref_audio_url": "https://example.com/reference.wav",
  "ref_text": "Optional transcript",
  "language": "vi",
  "num_step": 32,
  "speed": 1.1,
  "format": "mp3"
}
```

Required fields: `chunks`, `ref_audio_url`.

Defaults:

```text
language omitted / null
num_step=32
speed omitted / null
format=wav
```

`speed` is optional. Values greater than `1.0` produce faster, shorter speech; values below `1.0` produce slower, longer speech.

Supported formats: `wav`, `mp3`.

Response:

```json
{
  "request_id": "f5a4e...",
  "status": "queued",
  "created_at": 1778730000.0,
  "updated_at": 1778730000.0,
  "status_url": "/v1/tts/jobs/f5a4e...",
  "language": "vi",
  "input_chars": 840,
  "chunks_total": 5,
  "chunks_completed": 0,
  "chunks_failed": 0
}
```

Poll the job:

```http
GET /v1/tts/jobs/<request_id>
```

When the job succeeds, the status response includes:

```json
{
  "status": "succeeded",
  "audio_url": "/v1/tts/jobs/<request_id>/audio",
  "format": "mp3",
  "cache_hit": true,
  "transcript": "Reference transcript",
  "chunks_total": 5,
  "chunks_completed": 5,
  "chunks_failed": 0
}
```

Audio response is a **length-prefixed binary stream** (`application/octet-stream`). Each chunk is a complete audio file:

```text
[4 bytes: chunk_count uint32 BE]
[4 bytes: chunk_0_size uint32 BE][chunk_0 binary]
[4 bytes: chunk_1_size uint32 BE][chunk_1 binary]
...
```

Audio response headers:

```http
Content-Type: application/octet-stream
X-Request-Id: <uuid>
X-Chunks-Total: 3
X-Audio-Format: audio/wav or audio/mpeg
X-Cache-Hit: true|false
X-Transcript: <url-encoded-transcript>
X-Transcript-Encoding: urlencoded-utf8
```

Audio files are streamed in bounded blocks instead of reading whole chunks into memory. Tune block size with `OMNIVOICE_STREAM_CHUNK_SIZE_BYTES`, default `1048576`.

See `MIGRATION_CHUNKS_API.md` for full client integration guide with Node.js examples.

Generated job audio is deleted only after the server fully streams it to the client. If the client disconnects before the stream completes, the audio remains available for retry. Jobs that never complete a stream are kept for `OMNIVOICE_JOB_TTL_SECONDS`, default `3600`, and the cleanup batch runs every `OMNIVOICE_JOB_CLEANUP_INTERVAL_SECONDS`, default `600`.

Example:

```bash
JOB_JSON=$(curl -sS -X POST "$BASE_URL/v1/tts" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"chunks":["Xin chao","Day la test"],"ref_audio_url":"https://example.com/ref.wav","format":"mp3","num_step":16}')

STATUS_URL=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["status_url"])' <<< "$JOB_JSON")
curl -sS -H "Authorization: Bearer $API_TOKEN" "$BASE_URL$STATUS_URL"
```

### `POST /v1/tts/kokoro`

Kokoro uses the same async job shape as OmniVoice, but does not require reference audio. The request accepts `chunks`, not raw `text`.

Request:

```json
{
  "chunks": ["こんにちは。", "これはココロの日本語音声です。"],
  "language": "ja",
  "voice": "jf_alpha",
  "speed": 1.0,
  "format": "wav"
}
```

Required fields: `chunks`.

Defaults:

```text
language=auto/en-us -> Kokoro code a
voice omitted -> default voice for the resolved Kokoro language
speed=1.0
format=wav
```

Supported Kokoro language mappings:

```text
a/en-us -> af_heart
b/en-gb -> bf_emma
e/es    -> ef_dora
f/fr    -> ff_siwis
h/hi    -> hf_alpha
i/it    -> if_sara
j/ja    -> jf_alpha
p/pt    -> pf_dora
z/zh    -> zf_xiaobei
```

Submit and poll:

```bash
JOB_JSON=$(curl -sS -X POST "$BASE_URL/v1/tts/kokoro" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"chunks":["こんにちは。","これはココロの日本語音声です。"],"language":"ja","format":"wav"}')

STATUS_URL=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["status_url"])' <<< "$JOB_JSON")
curl -sS -H "Authorization: Bearer $API_TOKEN" "$BASE_URL$STATUS_URL"
```

When the job succeeds, download the audio stream from `audio_url`. It uses the same length-prefixed binary stream format as `/v1/tts`.

Kokoro preloads and warms up all supported language pipelines by default. Set `KOKORO_PRELOAD_LANGUAGES=j,z` to warm only selected languages, or keep `a,b,e,f,h,i,j,p,z` for full coverage.

## Cache

Set `OMNIVOICE_CACHE_DIR`, default `/ephemeral/omnivoice-cache`.

The service caches:

```text
ref-audio/<sha256(ref_audio_url)>.<ext>
transcripts/<sha256(ref_audio_url)>.json
tmp/
```

If `ref_text` is provided, it is used and written to transcript cache. If `ref_text` is missing, the server reuses a cached transcript when present. Otherwise, OmniVoice auto-transcribes the reference audio with Whisper during prompt creation and the resolved transcript is cached afterward.

## Chunked TTS Scheduler

`/v1/tts` and `/v1/tts/kokoro` both create one logical TTS job from caller-provided chunks. OmniVoice creates one `voice_clone_prompt` for the job and generates compatible chunks in GPU batches. Kokoro uses its own async worker path with `KOKORO_CONCURRENCY`; keep Kokoro at concurrency `1` because its shared misaki/phonemizer pipeline is not safe under concurrent calls.

```text
OMNIVOICE_BATCH_SIZE=12
OMNIVOICE_BUSY_BACKLOG_CHUNKS=24
KOKORO_CONCURRENCY=1
KOKORO_BUSY_BACKLOG_CHUNKS=24
OMNIVOICE_BATCH_MAX_WAIT_MS=50
OMNIVOICE_JOB_TTL_SECONDS=3600
OMNIVOICE_JOB_CLEANUP_INTERVAL_SECONDS=600
```

Client is responsible for splitting text into chunks before submitting. See `MIGRATION_CHUNKS_API.md` for splitting guidelines (recommended: 150–250 chars per chunk, split at sentence boundaries).

`OMNIVOICE_BATCH_SIZE` is the maximum number of chunks sent to one `model.generate(...)` call. Compatible chunks are grouped by `num_step`, `speed`, and `language`.

## TTS Language Handling

The TTS service runs the official `k2-fsa/OmniVoice` checkpoint directly. It does not load language-specific finetunes, auto-detect request text, or switch models per batch.

Explicit request `language` values are still normalized before generation. `zh`, `cmn`, and Mandarin-oriented `zh-*` aliases map to OmniVoice `zh`; explicit Cantonese aliases such as `yue` and `zh-hk` map to OmniVoice `yue`. `fr`, `fra`, and common `fr-*` aliases map to OmniVoice `fr`. `ar`, `ara`, and common `ar-*` aliases map to OmniVoice `arb`; explicit Arabic dialect IDs from `languages.md` such as `ars`, `ary`, `arz`, `afb`, and `apc` are preserved.

If `language` is omitted or set to `auto`, the server passes no language override to OmniVoice.

`OMNIVOICE_BUSY_BACKLOG_CHUNKS` is the OmniVoice admission gate. `KOKORO_BUSY_BACKLOG_CHUNKS` is the separate Kokoro admission gate. New requests return `429 Too Many Requests` with `Retry-After: 1` only when the same provider's queued+running chunks are already at its limit, so an OmniVoice backlog does not block `/v1/tts/kokoro`, and Kokoro backlog does not block `/v1/tts`. Accepted jobs are not rejected internally; they run to `succeeded` or `failed`.

`OMNIVOICE_BATCH_MAX_WAIT_MS` is only a short wait to let a partial batch fill with nearby traffic. It is not a public request queue.

## Acceleration

Set the GPU profile and model dtype:

```text
OMNIVOICE_GPU_PROFILE=auto
OMNIVOICE_DTYPE=fp16
```

Supported profiles are `auto`, `a100`, `h100`, and `generic`. Supported dtypes are `fp16` and `bf16`. A100 and H100 both support BF16, so benchmark both `fp16` and `bf16`; the faster path depends on the OmniVoice and `omnivoice-triton` backend path.

The official checkpoint remains the default:

```text
OMNIVOICE_MODEL=k2-fsa/OmniVoice
```

You can also benchmark the BF16-converted checkpoint as a drop-in model option:

```text
OMNIVOICE_MODEL=drbaph/OmniVoice-bf16
OMNIVOICE_DTYPE=bf16
```

Set `OMNIVOICE_ACCELERATION`:

```text
base
triton
hybrid
```

`triton` is the default TTS runtime. `base` is the official OmniVoice path for fallback/comparison. `triton` uses `omnivoice-triton` kernel fusion without CUDA Graph capture, so it is the safer optimized mode to test with chunk batching. `hybrid` uses CUDA Graph + Triton and should stay experimental; start with a small `OMNIVOICE_BATCH_SIZE`.

Recommended testing order:

```text
base   -> stable baseline
triton -> safer optimization test
hybrid -> single-flight experiment only
```

## Thunder Setup

Recommended instance:

```text
Mode: Production
GPU: NVIDIA A100 80GB
Count: 1 GPU
CPU/RAM: 15 vCPU / 120GB RAM
Primary Storage: 150-200GB
Ephemeral Storage: 100-300GB
Template: base / PyTorch
```

Clone this repo on Thunder:

```bash
git clone git@github.com:<your-org-or-user>/omnivoice-api.git
cd omnivoice-api
./scripts/install_thunder.sh
```

This installs Python deps into `~/venvs/omnivoice-api`, ensures Python `3.12+`, installs system packages, and installs NVIDIA drivers if `nvidia-smi` is not working.

```text
ffmpeg
tmux
curl
ca-certificates
lsb-release
build-essential
gcc
g++
ufw
NVIDIA driver packages when needed
```

Install options:

```bash
# Skip NVIDIA driver install/check
INSTALL_NVIDIA_DRIVER=0 ./scripts/install_thunder.sh

# Force a specific Ubuntu NVIDIA driver package, e.g. nvidia-driver-550
NVIDIA_DRIVER_VERSION=550 ./scripts/install_thunder.sh

# Use an existing Python >=3.12 binary
PYTHON_BIN=/usr/bin/python3.12 ./scripts/install_thunder.sh

# Disable deadsnakes fallback if Python 3.12 is missing from apt
ALLOW_DEADSNAKES_PPA=0 ./scripts/install_thunder.sh

# Add UFW allow rules and enable firewall non-interactively
ENABLE_UFW=1 ./scripts/install_thunder.sh

# Custom app/SSH ports for UFW rules
APP_PORT=8001 SSH_PORT=22 ./scripts/install_thunder.sh
```

If the script installs an NVIDIA driver, restart/reboot the instance before expecting `nvidia-smi` to work.

The install script adds UFW allow rules for SSH and the app port. By default it does not enable UFW. To enable manually:

```bash
sudo ufw allow ssh
sudo ufw allow 8001/tcp
sudo ufw enable
sudo ufw status verbose
```

Configure env:

```bash
cp .env.example .env
vim .env
```

Manual run:

```bash
source ~/venvs/omnivoice-api/bin/activate
set -a
source .env
set +a
uvicorn app:app --host 0.0.0.0 --port 8001
```

Expose through Thunder:

```bash
tnr ports forward 0 --add 8001
```

## Run With tmux

Some Thunder environments do not boot with `systemd` as PID 1. Use `tmux` so the API keeps running after you close SSH/CLI.

Quick start:

```bash
cd ~/New-project
./scripts/run_tmux.sh
```

Restart the existing tmux service:

```bash
cd ~/New-project
RESTART=1 ./scripts/run_tmux.sh
```

Use a different port:

```bash
cd ~/New-project
PORT=8081 ./scripts/run_tmux.sh
```

Install tmux:

```bash
sudo apt-get update
sudo apt-get install -y tmux
```

Start the API:

```bash
tmux new -s omnivoice
```

Inside tmux:

```bash
cd ~/New-project
source ~/venvs/omnivoice-api/bin/activate
set -a
source .env
set +a
uvicorn app:app --host 0.0.0.0 --port 8001
```

Detach while leaving the API running:

```text
Ctrl+B
D
```

Reattach:

```bash
tmux attach -t omnivoice
```

List sessions:

```bash
tmux ls
```

Stop the API:

```bash
tmux attach -t omnivoice
# then press Ctrl+C
exit
```

Expose the port:

```bash
tnr ports forward 0 --add 8001
```

Health check:

```bash
curl http://127.0.0.1:8001/health
```

## systemd Optional

Install the unit after editing `WorkingDirectory`, `EnvironmentFile`, and `ExecStart` if your paths differ:

```bash
sudo cp deploy/systemd/omnivoice-api.service /etc/systemd/system/omnivoice-api.service
sudo systemctl daemon-reload
sudo systemctl enable omnivoice-api
sudo systemctl start omnivoice-api
sudo systemctl status omnivoice-api
```

Update deployment:

```bash
git pull
source ~/venvs/omnivoice-api/bin/activate
pip install -r requirements.txt
sudo systemctl restart omnivoice-api
```

## Smoke Test Without GPU

For local API shape testing only:

```bash
OMNIVOICE_SKIP_MODEL_LOAD=1 API_TOKEN=test ./scripts/run_local.sh
```

The service returns silent WAV audio in this mode. MP3 still requires `ffmpeg`.

## Benchmark

Run from your local machine or from Thunder:

```bash
python scripts/benchmark_tts.py \
  --base-url "https://<instance-uuid>-8001.thundercompute.net" \
  --token "change-me" \
  --ref-audio-url "https://example.com/ref.wav" \
  --ref-audio-variants 1 \
  --text "Xin chao, day la benchmark OmniVoice." \
  --language vi \
  --requests 20 \
  --concurrency 4 \
  --speed 1.1 \
  --format mp3 \
  --results-json benchmark.json \
  --results-csv benchmark.csv
```

Use `--text-repeat` to create a long logical TTS job without creating a separate text file:

```bash
python scripts/benchmark_tts.py \
  --base-url "$BASE_URL" \
  --token "$API_TOKEN" \
  --ref-audio-url "$REF_AUDIO_URL" \
  --text "Day la cau benchmark cho long-form TTS." \
  --text-repeat 80 \
  --requests 1 \
  --concurrency 1 \
  --format mp3 \
  --results-json benchmark-long.json
```

To benchmark cold/warm behavior across many reference-audio cache keys while downloading the same source object, create query-string variants:

```bash
python scripts/benchmark_tts.py \
  --base-url "$BASE_URL" \
  --token "$API_TOKEN" \
  --ref-audio-url "https://example.com/ref.mp3" \
  --ref-audio-variants 8 \
  --ref-audio-variant-param a \
  --ref-audio-selection round-robin \
  --requests 40 \
  --concurrency 8 \
  --format mp3
```

This sends `ref.mp3?a=1`, `ref.mp3?a=2`, ... `ref.mp3?a=8`. Use `--ref-audio-selection grouped` if you want each variant to receive a contiguous block of requests instead of round-robin traffic.

Benchmark results include submit latency, end-to-end latency, status counts, accepted/rejected counts, chunk progress, audio duration, and RTF (`end_to_end_ms / audio_duration_ms`) when `ffprobe` is available. Install `ffmpeg` on the benchmark client for MP3 duration probing.

Compare WAV and MP3 with the same payload:

```bash
python scripts/benchmark_tts.py --base-url "$BASE_URL" --token "$API_TOKEN" --ref-audio-url "$REF_AUDIO_URL" --requests 20 --concurrency 4 --format wav
python scripts/benchmark_tts.py --base-url "$BASE_URL" --token "$API_TOKEN" --ref-audio-url "$REF_AUDIO_URL" --requests 20 --concurrency 4 --format mp3
```

The server-side matrix defaults to `triton` acceleration and includes both short and long logical TTS jobs. Set `MATRIX_ACCELERATIONS="base triton"` when you explicitly want to compare base vs Triton. The matrix value named `MATRIX_CONCURRENCIES` is kept for compatibility with old reports; it now sets both benchmark client concurrency and `OMNIVOICE_BATCH_SIZE`.

```text
MATRIX_TEXT_REPEATS="1 20"
MATRIX_BUSY_BACKLOG_MULTIPLIERS="2 4 8"
```

`1` benchmarks short text. Values greater than `1` repeat `TEXT` inside each request, so the report exercises chunk split, chunk batching, merge, and final-file behavior.

Busy backlog is swept as `OMNIVOICE_BATCH_SIZE * MATRIX_BUSY_BACKLOG_MULTIPLIERS`. Set `MATRIX_BUSY_BACKLOG_CHUNKS="32 64 128"` when you want exact backlog values instead. The report includes `Accepted`, `429`, and `Non-429 failed`, plus a `Suggested configs` section that picks the best short no-429, long full-accept, and long backpressure tradeoff cases.

Run the server-side matrix benchmark when comparing A100 and H100:

```bash
# Auto-detects A100/H100 from nvidia-smi and chooses the matching matrix.
./scripts/run_server_benchmark_report.sh

# Run one-ref, eight-ref, and one-ref-per-request cases derived from one .mp3.
MATRIX_REF_AUDIO_VARIANTS="1 8 requests" REF_AUDIO_VARIANT_PARAM=a ./scripts/run_server_benchmark_report.sh

# Force the A100 matrix.
GPU_PROFILE=a100 ./scripts/run_server_benchmark_report.sh

# Force the H100 matrix with larger chunk batches.
GPU_PROFILE=h100 ./scripts/run_server_benchmark_report.sh

# Compare official and BF16-converted checkpoints.
MATRIX_MODELS="k2-fsa/OmniVoice drbaph/OmniVoice-bf16" MATRIX_DTYPES="bf16" ./scripts/run_server_benchmark_report.sh
```

H100 defaults are:

```text
MATRIX_CONCURRENCIES="8 12 16 24 32"
MATRIX_DTYPES="fp16"
MATRIX_STEPS="16 32"
MATRIX_TEXT_REPEATS="1 20"
MATRIX_BUSY_BACKLOG_MULTIPLIERS="2 4 8"
OMNIVOICE_BATCH_MAX_WAIT_MS=50
```

For a quick H100 smoke test:

```bash
REQUESTS=2 GPU_PROFILE=h100 MATRIX_ACCELERATIONS=base MATRIX_CONCURRENCIES=16 MATRIX_DTYPES=fp16 MATRIX_SPEEDS=1.1 ./scripts/run_server_benchmark_report.sh
```

A100 defaults are:

```text
MATRIX_CONCURRENCIES="4 6 8 12"
MATRIX_DTYPES="fp16"
MATRIX_STEPS="16 32"
MATRIX_TEXT_REPEATS="1 20"
MATRIX_BUSY_BACKLOG_MULTIPLIERS="2 4 8"
OMNIVOICE_BATCH_MAX_WAIT_MS=100
```
