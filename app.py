import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from urllib.parse import urlparse

import httpx
import numpy as np
import soundfile as sf
from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi import Response
from fastapi.responses import FileResponse
from pydantic import BaseModel


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("omnivoice-api")


MODEL_NAME = os.getenv("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
ASR_MODEL_NAME = os.getenv("OMNIVOICE_ASR_MODEL", "openai/whisper-large-v3-turbo")
DEVICE_MAP = os.getenv("OMNIVOICE_DEVICE", "cuda:0")
API_TOKEN = os.getenv("API_TOKEN", "")
CACHE_DIR = Path(os.getenv("OMNIVOICE_CACHE_DIR", "/ephemeral/omnivoice-cache"))
MAX_TEXT_CHARS = int(os.getenv("OMNIVOICE_MAX_TEXT_CHARS", "3000"))
MAX_CONCURRENCY = int(os.getenv("OMNIVOICE_CONCURRENCY", "4"))
DOWNLOAD_TIMEOUT = float(os.getenv("OMNIVOICE_DOWNLOAD_TIMEOUT", "60"))
REQUEST_TIMEOUT = float(os.getenv("OMNIVOICE_REQUEST_TIMEOUT", "300"))
SKIP_MODEL_LOAD = os.getenv("OMNIVOICE_SKIP_MODEL_LOAD", "0") == "1"
ACCELERATION = os.getenv("OMNIVOICE_ACCELERATION", "base").lower().strip()

SUPPORTED_FORMATS = {"wav", "mp3"}
SUPPORTED_ACCELERATIONS = {"base", "hybrid"}
REF_AUDIO_DIR = CACHE_DIR / "ref-audio"
TRANSCRIPT_DIR = CACHE_DIR / "transcripts"
TMP_DIR = CACHE_DIR / "tmp"


app = FastAPI(title="OmniVoice API", version="1.0.0")
model = None
model_loaded = False
generation_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
active_requests = 0
active_generations = 0
queued_generations = 0
metrics_lock = asyncio.Lock()
cache_locks: dict[str, asyncio.Lock] = {}
cache_locks_guard = asyncio.Lock()


class TTSRequest(BaseModel):
    text: str
    ref_audio_url: str
    ref_text: Optional[str] = None
    language: Optional[str] = None
    num_step: int = 32
    format: str = "wav"


@dataclass
class ReferenceCacheEntry:
    audio_path: Path
    transcript: Optional[str]
    audio_cache_hit: bool


def _ensure_dirs() -> None:
    REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_audio_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}:
        return suffix
    return ".audio"


def _audio_cache_path(ref_audio_url: str) -> Path:
    return REF_AUDIO_DIR / f"{_sha256(ref_audio_url)}{_safe_audio_suffix(ref_audio_url)}"


def _transcript_cache_path(ref_audio_url: str) -> Path:
    return TRANSCRIPT_DIR / f"{_sha256(ref_audio_url)}.json"


def _read_transcript(ref_audio_url: str) -> Optional[str]:
    path = _transcript_cache_path(ref_audio_url)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        transcript = data.get("transcript")
        if isinstance(transcript, str) and transcript.strip():
            return transcript.strip()
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read transcript cache %s: %s", path, exc)
    return None


def _write_transcript(ref_audio_url: str, transcript: str) -> None:
    transcript = transcript.strip()
    if not transcript:
        return
    path = _transcript_cache_path(ref_audio_url)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = {
        "ref_audio_url": ref_audio_url,
        "transcript": transcript,
    }
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


async def _get_cache_lock(key: str) -> asyncio.Lock:
    async with cache_locks_guard:
        lock = cache_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            cache_locks[key] = lock
        return lock


async def _download_ref_audio(ref_audio_url: str, target: Path) -> None:
    tmp_target = target.with_suffix(f"{target.suffix}.tmp")
    async with httpx.AsyncClient(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT) as client:
        async with client.stream("GET", ref_audio_url) as response:
            if response.status_code >= 400:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not download ref_audio_url: HTTP {response.status_code}",
                )
            with tmp_target.open("wb") as handle:
                async for chunk in response.aiter_bytes():
                    handle.write(chunk)
    tmp_target.replace(target)


async def _resolve_reference(req: TTSRequest) -> ReferenceCacheEntry:
    cache_key = _sha256(req.ref_audio_url)
    lock = await _get_cache_lock(cache_key)
    async with lock:
        audio_path = _audio_cache_path(req.ref_audio_url)
        audio_cache_hit = audio_path.exists()
        if not audio_cache_hit:
            await _download_ref_audio(req.ref_audio_url, audio_path)

        if req.ref_text and req.ref_text.strip():
            transcript = req.ref_text.strip()
            _write_transcript(req.ref_audio_url, transcript)
        else:
            transcript = _read_transcript(req.ref_audio_url)

        return ReferenceCacheEntry(
            audio_path=audio_path,
            transcript=transcript,
            audio_cache_hit=audio_cache_hit,
        )


def _load_model() -> None:
    global model, model_loaded
    if SKIP_MODEL_LOAD:
        logger.warning("OMNIVOICE_SKIP_MODEL_LOAD=1; model loading is disabled.")
        model_loaded = False
        return
    if ACCELERATION not in SUPPORTED_ACCELERATIONS:
        raise ValueError(
            "OMNIVOICE_ACCELERATION must be one of: "
            f"{', '.join(sorted(SUPPORTED_ACCELERATIONS))}"
        )

    import torch

    logger.info(
        "Loading OmniVoice model=%s device=%s acceleration=%s ...",
        MODEL_NAME,
        DEVICE_MAP,
        ACCELERATION,
    )
    if ACCELERATION == "hybrid":
        from omnivoice_triton import create_runner

        runner = create_runner(
            "hybrid",
            device=DEVICE_MAP,
            model_id=MODEL_NAME,
            dtype="fp16",
        )
        runner.load_model()
        runner.model.load_asr_model(model_name=ASR_MODEL_NAME)
        model = runner.model
    else:
        from omnivoice import OmniVoice

        model = OmniVoice.from_pretrained(
            MODEL_NAME,
            device_map=DEVICE_MAP,
            dtype=torch.float16,
            load_asr=True,
            asr_model_name=ASR_MODEL_NAME,
        )
    model_loaded = True
    logger.info("OmniVoice model loaded with acceleration=%s.", ACCELERATION)


def _generate_wav(req: TTSRequest, ref: ReferenceCacheEntry, output_path: Path) -> str:
    if model is None:
        raise RuntimeError("OmniVoice model is not loaded.")

    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref.audio_path),
        ref_text=ref.transcript,
    )
    resolved_transcript = voice_clone_prompt.ref_text
    if resolved_transcript:
        _write_transcript(req.ref_audio_url, resolved_transcript)

    audios = model.generate(
        text=req.text.strip(),
        language=req.language or None,
        voice_clone_prompt=voice_clone_prompt,
        num_step=req.num_step,
    )
    sf.write(str(output_path), audios[0], model.sampling_rate)
    return resolved_transcript or ""


def _write_silent_test_wav(output_path: Path) -> str:
    waveform = np.zeros(2400, dtype=np.float32)
    sf.write(str(output_path), waveform, 24000)
    return ""


def _convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for format=mp3 but was not found.")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(mp3_path),
        ],
        check=True,
    )


async def _metric_delta(name: str, delta: int) -> None:
    global active_requests, active_generations, queued_generations
    async with metrics_lock:
        if name == "active_requests":
            active_requests += delta
        elif name == "active_generations":
            active_generations += delta
        elif name == "queued_generations":
            queued_generations += delta
        else:
            raise ValueError(f"Unknown metric: {name}")


async def _request_delta(delta: int) -> None:
    await _metric_delta("active_requests", delta)


async def _active_generation_delta(delta: int) -> None:
    await _metric_delta("active_generations", delta)


async def _queued_generation_delta(delta: int) -> None:
    await _metric_delta("queued_generations", delta)


async def _get_runtime_metrics() -> dict[str, int]:
    async with metrics_lock:
        return {
            "active_requests": active_requests,
            "active_generations": active_generations,
            "queued_generations": queued_generations,
            "max_concurrency": MAX_CONCURRENCY,
            "available_generation_slots": max(MAX_CONCURRENCY - active_generations, 0),
        }


async def _generate_response_audio(
    req: TTSRequest,
    ref: ReferenceCacheEntry,
    output_wav: Path,
    cleanup_paths: list[Path],
) -> tuple[Path, str, str]:
    transcript = ref.transcript or ""

    await _queued_generation_delta(1)
    try:
        await generation_semaphore.acquire()
    finally:
        await _queued_generation_delta(-1)

    await _active_generation_delta(1)
    try:
        if SKIP_MODEL_LOAD:
            _write_silent_test_wav(output_wav)
        else:
            transcript = await asyncio.to_thread(_generate_wav, req, ref, output_wav)
    finally:
        await _active_generation_delta(-1)
        generation_semaphore.release()

    if req.format == "mp3":
        output_mp3 = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=TMP_DIR).name)
        cleanup_paths.append(output_mp3)
        await asyncio.to_thread(_convert_wav_to_mp3, output_wav, output_mp3)
        return output_mp3, "audio/mpeg", transcript

    return output_wav, "audio/wav", transcript


def _remove_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove temp file %s: %s", path, exc)


def _validate_token(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="API_TOKEN is not configured.")
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _validate_request(req: TTSRequest) -> None:
    req.format = req.format.lower().strip()
    if req.format not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="format must be wav or mp3.")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required.")
    if len(req.text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"text exceeds OMNIVOICE_MAX_TEXT_CHARS={MAX_TEXT_CHARS}.",
        )
    if not req.ref_audio_url or not req.ref_audio_url.strip():
        raise HTTPException(status_code=400, detail="ref_audio_url is required.")
    parsed = urlparse(req.ref_audio_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="ref_audio_url must be http(s).")
    if req.num_step < 4 or req.num_step > 64:
        raise HTTPException(status_code=400, detail="num_step must be between 4 and 64.")


@app.on_event("startup")
async def startup() -> None:
    _ensure_dirs()
    await asyncio.to_thread(_load_model)


@app.get("/health")
async def health() -> dict[str, object]:
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = None

    runtime_metrics = await _get_runtime_metrics()
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "gpu": gpu,
        "cache_audio_count": len(list(REF_AUDIO_DIR.glob("*"))),
        "cache_transcript_count": len(list(TRANSCRIPT_DIR.glob("*.json"))),
        "acceleration": ACCELERATION,
        **runtime_metrics,
    }


@app.post("/v1/tts")
async def tts(
    req: TTSRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(_validate_token),
) -> Response:
    await _request_delta(1)
    try:
        return await _tts(req, background_tasks)
    finally:
        await _request_delta(-1)


async def _tts(req: TTSRequest, background_tasks: BackgroundTasks) -> Response:
    _validate_request(req)
    request_id = str(uuid.uuid4())
    ref = await _resolve_reference(req)

    output_wav = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TMP_DIR).name)
    cleanup_paths = [output_wav]

    try:
        output_path, media_type, transcript = await asyncio.wait_for(
            _generate_response_audio(req, ref, output_wav, cleanup_paths),
            timeout=REQUEST_TIMEOUT,
        )
    except TimeoutError as exc:
        _remove_files(cleanup_paths)
        raise HTTPException(status_code=504, detail="TTS request timed out.") from exc
    except HTTPException:
        _remove_files(cleanup_paths)
        raise
    except Exception as exc:
        _remove_files(cleanup_paths)
        logger.exception("TTS request %s failed", request_id)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc

    background_tasks.add_task(_remove_files, cleanup_paths)
    headers = {
        "X-Request-Id": request_id,
        "X-Cache-Hit": str(ref.audio_cache_hit).lower(),
        "X-Transcript": quote(transcript or "", safe=""),
        "X-Transcript-Encoding": "urlencoded-utf8",
    }
    filename = f"{request_id}.{req.format}"
    return FileResponse(
        path=str(output_path),
        media_type=media_type,
        filename=filename,
        headers=headers,
        background=background_tasks,
    )
