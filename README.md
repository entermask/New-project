# OmniVoice API

Standalone FastAPI wrapper for [OmniVoice](https://github.com/k2-fsa/OmniVoice), intended to run on a Thunder Compute GPU instance and serve an existing TTS queue through a single `/v1/tts` endpoint.

## API

### `GET /health`

Returns model, GPU, concurrency, and cache status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "gpu": "NVIDIA A100 80GB",
  "cache_audio_count": 12,
  "cache_transcript_count": 12,
  "active_requests": 3,
  "active_generations": 1,
  "queued_generations": 2,
  "max_concurrency": 4,
  "available_generation_slots": 3,
  "acceleration": "base"
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
  "text": "Noi dung can doc",
  "ref_audio_url": "https://example.com/reference.wav",
  "ref_text": "Optional transcript",
  "language": "vi",
  "num_step": 32,
  "speed": 1.1,
  "format": "mp3"
}
```

Required fields: `text`, `ref_audio_url`.

Defaults:

```text
language omitted / null
num_step=32
speed omitted / null
format=wav
```

`speed` is optional. Values greater than `1.0` produce faster, shorter speech; values below `1.0` produce slower, longer speech.

Supported formats: `wav`, `mp3`.

Response body is binary audio. Headers:

```http
Content-Type: audio/wav or audio/mpeg
X-Request-Id: <uuid>
X-Cache-Hit: true|false
X-Transcript: <url-encoded-transcript>
X-Transcript-Encoding: urlencoded-utf8
```

## Cache

Set `OMNIVOICE_CACHE_DIR`, default `/ephemeral/omnivoice-cache`.

The service caches:

```text
ref-audio/<sha256(ref_audio_url)>.<ext>
transcripts/<sha256(ref_audio_url)>.json
tmp/
```

If `ref_text` is provided, it is used and written to transcript cache. If `ref_text` is missing, the server reuses a cached transcript when present. Otherwise, OmniVoice auto-transcribes the reference audio with Whisper during prompt creation and the resolved transcript is cached afterward.

## Acceleration

Set `OMNIVOICE_ACCELERATION`:

```text
base
hybrid
```

`base` is the official OmniVoice path and is the default. `hybrid` uses `omnivoice-triton` CUDA Graph + Triton kernel optimization for experimental benchmarking. Because CUDA Graph uses captured static buffers, test concurrency carefully before using `hybrid` for live traffic.

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

## systemd

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
  --text "Xin chao, day la benchmark OmniVoice." \
  --language vi \
  --requests 20 \
  --concurrency 4 \
  --speed 1.1 \
  --format mp3 \
  --results-json benchmark.json \
  --results-csv benchmark.csv
```

Compare WAV and MP3 with the same payload:

```bash
python scripts/benchmark_tts.py --base-url "$BASE_URL" --token "$API_TOKEN" --ref-audio-url "$REF_AUDIO_URL" --requests 20 --concurrency 4 --format wav
python scripts/benchmark_tts.py --base-url "$BASE_URL" --token "$API_TOKEN" --ref-audio-url "$REF_AUDIO_URL" --requests 20 --concurrency 4 --format mp3
```

For hybrid acceleration, start with `OMNIVOICE_CONCURRENCY=1`, then benchmark `--concurrency 1`, `2`, and `4`.
