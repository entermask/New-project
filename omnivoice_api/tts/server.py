import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import OrderedDict
from collections import deque
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
from fastapi import UploadFile, File, Form
from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException

from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
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
DOWNLOAD_TIMEOUT = float(os.getenv("OMNIVOICE_DOWNLOAD_TIMEOUT", "60"))
REQUEST_TIMEOUT = float(os.getenv("OMNIVOICE_REQUEST_TIMEOUT", "300"))
JOB_TTL_SECONDS = int(os.getenv("OMNIVOICE_JOB_TTL_SECONDS", "3600"))
SKIP_MODEL_LOAD = os.getenv("OMNIVOICE_SKIP_MODEL_LOAD", "0") == "1"
GPU_PROFILE = os.getenv("OMNIVOICE_GPU_PROFILE", "auto").lower().strip()
MODEL_DTYPE = os.getenv("OMNIVOICE_DTYPE", "fp16").lower().strip()
ACCELERATION = os.getenv("OMNIVOICE_ACCELERATION", "base").lower().strip()

BATCH_SIZE = max(1, int(os.getenv("OMNIVOICE_BATCH_SIZE", "12")))
BATCH_MAX_WAIT_MS = max(0.0, float(os.getenv("OMNIVOICE_BATCH_MAX_WAIT_MS", "100")))
BUSY_BACKLOG_CHUNKS = max(1, int(os.getenv("OMNIVOICE_BUSY_BACKLOG_CHUNKS", str(BATCH_SIZE * 2))))
PROMPT_CACHE_SIZE = max(1, int(os.getenv("OMNIVOICE_PROMPT_CACHE_SIZE", "64")))
JOB_CLEANUP_INTERVAL_SECONDS = 60
STREAM_CHUNK_SIZE_BYTES = max(16 * 1024, int(os.getenv("OMNIVOICE_STREAM_CHUNK_SIZE_BYTES", str(1024 * 1024))))

SUPPORTED_FORMATS = {"wav", "mp3"}
SUPPORTED_ACCELERATIONS = {"base", "triton", "hybrid"}
SUPPORTED_GPU_PROFILES = {"auto", "generic", "a100", "h100"}
SUPPORTED_MODEL_DTYPES = {"fp16", "bf16"}
REF_AUDIO_DIR = CACHE_DIR / "ref-audio"
TRANSCRIPT_DIR = CACHE_DIR / "transcripts"
TMP_DIR = CACHE_DIR / "tmp"
JOB_DIR = CACHE_DIR / "jobs"


app = FastAPI(title="OmniVoice TTS API", version="1.0.0")
model = None
model_runner = None
model_loaded = False
resolved_gpu_name: Optional[str] = None
resolved_gpu_profile: Optional[str] = None
resolved_model_dtype: Optional[str] = None
generation_queue: asyncio.Queue["ChunkGenerationItem"] = asyncio.Queue()
generation_batch_worker_task: Optional[asyncio.Task] = None
gpu_inference_semaphore = asyncio.Semaphore(1)
gpu_generation_semaphore = gpu_inference_semaphore
deferred_generation_items: deque["ChunkGenerationItem"] = deque()
job_cleanup_task: Optional[asyncio.Task] = None
active_requests = 0
active_generations = 0
active_generation_batches = 0
queued_generations = 0
metrics_lock = asyncio.Lock()
cache_locks: dict[str, asyncio.Lock] = {}
cache_locks_guard = asyncio.Lock()
tts_jobs: dict[str, "TTSJob"] = {}
tts_jobs_lock = asyncio.Lock()
voice_prompt_cache: OrderedDict[str, object] = OrderedDict()
voice_prompt_cache_locks: dict[str, asyncio.Lock] = {}
voice_prompt_cache_guard = asyncio.Lock()


class TTSRequest(BaseModel):
    chunks: list[str]
    ref_audio_url: str
    ref_text: str
    language: Optional[str] = None
    num_step: int = 32
    speed: Optional[float] = None
    guidance_scale: float = 2.0
    format: str = "wav"


@dataclass
class ReferenceCacheEntry:
    audio_path: Path
    transcript: Optional[str]
    audio_cache_hit: bool


@dataclass
class ChunkGenerationItem:
    request_id: str
    chunk_index: int
    text: str
    req: TTSRequest
    voice_clone_prompt: object
    output_wav: Path
    future: asyncio.Future[str]



@dataclass
class TTSJob:
    request_id: str
    status: str
    created_at: float
    updated_at: float
    format: str
    language: Optional[str] = None
    detail: Optional[str] = None
    chunk_paths: Optional[list[Path]] = None
    chunk_media_type: Optional[str] = None
    transcript: str = ""
    audio_cache_hit: Optional[bool] = None
    cleanup_paths: Optional[list[Path]] = None
    chunks_total: int = 0
    chunks_completed: int = 0
    chunks_failed: int = 0
    input_chars: int = 0


def _ensure_dirs() -> None:
    REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)


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


AUTO_LANGUAGE_VALUES = {"", "auto", "detect", "unknown", "none", "null"}
ZH_ALIASES = {
    "zh",
    "zh-cn",
    "zh-hans",
    "zh-sg",
    "zh-my",
    "zh-tw",
    "cmn",
    "zho",
    "chi",
    "chinese",
    "mandarin",
}
YUE_ALIASES = {"yue", "zh-hk", "zh-mo", "cantonese"}
FR_ALIASES = {"fr", "fr-fr", "fr-ca", "fra", "fre", "french"}
AR_STANDARD_ALIASES = {"ar", "ara", "arabic", "standard-arabic", "msa"}
ARABIC_LANGUAGE_IDS = {
    "aao",
    "abv",
    "acm",
    "acw",
    "acx",
    "aec",
    "adf",
    "afb",
    "aju",
    "ajp",
    "apc",
    "apd",
    "arb",
    "arq",
    "ars",
    "ary",
    "arz",
    "avl",
    "ayl",
    "ayp",
    "aeb",
    "shu",
}
def _normalize_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    value = language.strip().lower().replace("_", "-")
    if value in AUTO_LANGUAGE_VALUES:
        return None
    if value in YUE_ALIASES:
        return "yue"
    if value in ZH_ALIASES:
        return "zh"
    if value in FR_ALIASES:
        return "fr"
    if value in AR_STANDARD_ALIASES:
        return "arb"
    if value in ARABIC_LANGUAGE_IDS:
        return value
    if value.startswith("zh-"):
        return "zh"
    if value.startswith("fr-"):
        return "fr"
    if value.startswith("ar-"):
        return "arb"
    return value


async def _get_cache_lock(key: str) -> asyncio.Lock:
    async with cache_locks_guard:
        lock = cache_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            cache_locks[key] = lock
        return lock


def _voice_prompt_cache_key(req: TTSRequest, ref: ReferenceCacheEntry) -> str:
    transcript = (ref.transcript or "").strip()
    return _sha256(
        json.dumps(
            {
                "model": MODEL_NAME,
                "acceleration": ACCELERATION,
                "audio_path": str(ref.audio_path),
                "transcript": transcript,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )


async def _get_voice_prompt_cache_lock(key: str) -> asyncio.Lock:
    async with voice_prompt_cache_guard:
        lock = voice_prompt_cache_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            voice_prompt_cache_locks[key] = lock
        return lock


async def _get_voice_clone_prompt(req: TTSRequest, ref: ReferenceCacheEntry) -> object:
    cache_key = _voice_prompt_cache_key(req, ref)
    async with voice_prompt_cache_guard:
        cached = voice_prompt_cache.get(cache_key)
        if cached is not None:
            voice_prompt_cache.move_to_end(cache_key)
            return cached

    cache_lock = await _get_voice_prompt_cache_lock(cache_key)
    async with cache_lock:
        async with voice_prompt_cache_guard:
            cached = voice_prompt_cache.get(cache_key)
            if cached is not None:
                voice_prompt_cache.move_to_end(cache_key)
                return cached

        async with gpu_generation_semaphore:
            prompt = await _run_gpu_operation_with_oom_retry(
                "voice clone prompt",
                lambda: asyncio.to_thread(_create_voice_clone_prompt, req, ref),
            )

        async with voice_prompt_cache_guard:
            voice_prompt_cache[cache_key] = prompt
            voice_prompt_cache.move_to_end(cache_key)
            while len(voice_prompt_cache) > PROMPT_CACHE_SIZE:
                voice_prompt_cache.popitem(last=False)
        return prompt


async def _download_ref_audio(ref_audio_url: str, target: Path) -> None:
    tmp_target = target.with_suffix(f"{target.suffix}.tmp")
    last_exc = None
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=DOWNLOAD_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as client:
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
            return
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Attempt %d/3 to download ref audio failed: %s. Retrying in %ds...",
                attempt,
                exc,
                attempt,
            )
            if attempt < 3:
                await asyncio.sleep(attempt)
    
    raise HTTPException(
        status_code=400,
        detail=f"Could not download ref_audio_url after 3 attempts. Last error: {last_exc}",
    )


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


def _cuda_device_index() -> int:
    if DEVICE_MAP.startswith("cuda:"):
        try:
            return int(DEVICE_MAP.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def _gpu_profile_from_name(gpu_name: Optional[str]) -> str:
    name = (gpu_name or "").lower()
    if "h100" in name or "h200" in name:
        return "h100"
    if "a100" in name:
        return "a100"
    return "generic"


def _resolve_gpu_profile(gpu_name: Optional[str]) -> str:
    if GPU_PROFILE not in SUPPORTED_GPU_PROFILES:
        raise ValueError(
            "OMNIVOICE_GPU_PROFILE must be one of: "
            f"{', '.join(sorted(SUPPORTED_GPU_PROFILES))}"
        )
    if GPU_PROFILE == "auto":
        return _gpu_profile_from_name(gpu_name)
    return GPU_PROFILE


def _resolve_model_dtype() -> str:
    if MODEL_DTYPE not in SUPPORTED_MODEL_DTYPES:
        raise ValueError(
            "OMNIVOICE_DTYPE must be one of: "
            f"{', '.join(sorted(SUPPORTED_MODEL_DTYPES))}"
        )
    return MODEL_DTYPE


def _torch_dtype(torch_module: object, dtype_name: str) -> object:
    return torch_module.float16


def _clear_cuda_cache(context: str) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        gpu_index = _cuda_device_index()
        if gpu_index >= torch.cuda.device_count():
            return

        with torch.cuda.device(gpu_index):
            free_before, total = torch.cuda.mem_get_info()
            torch.cuda.empty_cache()
            free_after, _ = torch.cuda.mem_get_info()

        logger.info(
            "CUDA cache cleared after %s: free %.2f GiB -> %.2f GiB / %.2f GiB",
            context,
            free_before / 1024**3,
            free_after / 1024**3,
            total / 1024**3,
        )
    except Exception as exc:
        logger.warning("Could not clear CUDA cache after %s: %s", context, exc)


def _is_cuda_oom_error(exc: Exception) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass

    message = str(exc).lower()
    return (
        "cuda out of memory" in message
        or "outofmemoryerror" in message
        or "tried to allocate" in message
    )


async def _run_gpu_operation_with_oom_retry(context: str, operation):
    for attempt in range(2):
        try:
            return await operation()
        except Exception as exc:
            if not _is_cuda_oom_error(exc) or attempt == 1:
                raise
            logger.warning(
                "CUDA OOM during %s; clearing cache and retrying once: %s",
                context,
                exc,
            )
            _clear_cuda_cache(f"{context} OOM")

    raise RuntimeError(f"{context} failed after OOM retry.")


def _load_model() -> None:
    global model, model_loaded, model_runner
    global resolved_gpu_name, resolved_gpu_profile, resolved_model_dtype
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

    gpu_name = None
    if torch.cuda.is_available():
        gpu_index = _cuda_device_index()
        if gpu_index < torch.cuda.device_count():
            gpu_name = torch.cuda.get_device_name(gpu_index)
    gpu_profile = _resolve_gpu_profile(gpu_name)
    model_dtype = _resolve_model_dtype()

    logger.info(
        "Loading OmniVoice model=%s device=%s acceleration=%s gpu_profile=%s dtype=%s ...",
        MODEL_NAME,
        DEVICE_MAP,
        ACCELERATION,
        gpu_profile,
        model_dtype,
    )
    if ACCELERATION in {"triton", "hybrid"}:
        from omnivoice_triton import create_runner

        runner = create_runner(
            ACCELERATION,
            device=DEVICE_MAP,
            model_id=MODEL_NAME,
            dtype=model_dtype,
        )
        runner.load_model()
        model_runner = runner
        model = runner.model
    else:
        from omnivoice import OmniVoice

        model = OmniVoice.from_pretrained(
            MODEL_NAME,
            device_map=DEVICE_MAP,
            dtype=_torch_dtype(torch, model_dtype),
            load_asr=False,
        )
    resolved_gpu_name = gpu_name
    resolved_gpu_profile = gpu_profile
    resolved_model_dtype = model_dtype
    model_loaded = True
    logger.info(
        "OmniVoice model loaded with acceleration=%s gpu_profile=%s dtype=%s.",
        ACCELERATION,
        gpu_profile,
        model_dtype,
    )


def _generation_kwargs(texts: list[str], reqs: list[TTSRequest], voice_clone_prompts: list[object]) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "text": [t.strip() for t in texts],
        "language": [req.language or None for req in reqs],
        "voice_clone_prompt": voice_clone_prompts,
        "num_step": reqs[0].num_step,
        "guidance_scale": reqs[0].guidance_scale,
    }
    if any(req.speed is not None for req in reqs):
        kwargs["speed"] = [req.speed for req in reqs]
    return kwargs


def _create_voice_clone_prompt(req: TTSRequest, ref: ReferenceCacheEntry) -> object:
    if model is None:
        raise RuntimeError("OmniVoice model is not loaded.")

    ref_text = (ref.transcript or "").strip()
    ref_text = ref_text or (_read_transcript(req.ref_audio_url) or "").strip()
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref.audio_path),
        ref_text=ref_text or None,
    )
    resolved_transcript = (voice_clone_prompt.ref_text or "").strip()
    if resolved_transcript:
        _write_transcript(req.ref_audio_url, resolved_transcript)

    return voice_clone_prompt


def _generate_wav_batch(items: list[ChunkGenerationItem]) -> list[str]:
    if model is None:
        raise RuntimeError("OmniVoice model is not loaded.")

    logger.info(
        "Chunk batch generate: items=%d jobs=%d language=%s num_step=%d guidance_scale=%.2f",
        len(items),
        len({item.request_id for item in items}),
        items[0].req.language or "auto",
        items[0].req.num_step,
        items[0].req.guidance_scale,
    )

    gen_start = time.monotonic()
    texts = [item.text for item in items]
    prompts = [item.voice_clone_prompt for item in items]
    reqs = [item.req for item in items]
    transcripts = [prompt.ref_text or "" for prompt in prompts]
    audios = model.generate(**_generation_kwargs(texts, reqs, prompts))
    gen_elapsed = time.monotonic() - gen_start

    if len(audios) != len(items):
        raise RuntimeError(f"OmniVoice returned {len(audios)} audios for {len(items)} requests.")

    write_start = time.monotonic()
    for item, audio in zip(items, audios):
        sf.write(str(item.output_wav), audio, model.sampling_rate)
    write_elapsed = time.monotonic() - write_start

    logger.info(
        "Chunk batch done: items=%d generate=%.1fms write=%.1fms",
        len(items),
        gen_elapsed * 1000,
        write_elapsed * 1000,
    )
    return transcripts


def _write_silent_test_wav(output_path: Path) -> str:
    waveform = np.zeros(2400, dtype=np.float32)
    sf.write(str(output_path), waveform, 24000)
    return ""


def _convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for format=mp3 but was not found.")
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise RuntimeError(f"WAV file is missing or empty: {wav_path}")

    retry_timeouts = [60, 120, 180]  # 1m -> 2m -> 3m
    cmd = [
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
    ]

    for attempt, timeout_s in enumerate(retry_timeouts, 1):
        try:
            subprocess.run(cmd, check=True, timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            logger.warning(
                "ffmpeg WAV->MP3 timed out after %ds (attempt %d/%d): %s",
                timeout_s,
                attempt,
                len(retry_timeouts),
                wav_path,
            )
            # Clean up partial output before retry
            if mp3_path.exists():
                try:
                    mp3_path.unlink()
                except OSError:
                    pass
            if attempt >= len(retry_timeouts):
                raise RuntimeError(
                    f"ffmpeg WAV->MP3 conversion timed out after {len(retry_timeouts)} "
                    f"attempts (last timeout: {timeout_s}s): {wav_path}"
                )


async def _metric_delta(name: str, delta: int) -> None:
    global active_requests, active_generations, active_generation_batches, queued_generations
    async with metrics_lock:
        if name == "active_requests":
            active_requests += delta
        elif name == "active_generations":
            active_generations += delta
        elif name == "active_generation_batches":
            active_generation_batches += delta
        elif name == "queued_generations":
            queued_generations += delta
        else:
            raise ValueError(f"Unknown metric: {name}")


async def _request_delta(delta: int) -> None:
    await _metric_delta("active_requests", delta)


async def _active_generation_delta(delta: int) -> None:
    await _metric_delta("active_generations", delta)


async def _active_generation_batch_delta(delta: int) -> None:
    await _metric_delta("active_generation_batches", delta)


async def _queued_generation_delta(delta: int) -> None:
    await _metric_delta("queued_generations", delta)


async def _get_runtime_metrics() -> dict[str, object]:
    async with tts_jobs_lock:
        job_counts = {
            "queued": sum(1 for job in tts_jobs.values() if job.status == "queued"),
            "running": sum(1 for job in tts_jobs.values() if job.status == "running"),
            "succeeded": sum(1 for job in tts_jobs.values() if job.status == "succeeded"),
            "failed": sum(1 for job in tts_jobs.values() if job.status == "failed"),
        }
        active_tts_jobs = job_counts["queued"] + job_counts["running"]

    async with metrics_lock:
        outstanding_chunks = queued_generations + active_generations
        return {
            "active_requests": active_requests,
            "active_generations": active_generations,
            "active_generation_batches": active_generation_batches,
            "queued_generations": queued_generations,
            "queued_chunks": queued_generations,
            "running_chunks": active_generations,
            "outstanding_chunks": outstanding_chunks,
            "deferred_chunks": len(deferred_generation_items),
            "active_tts_jobs": active_tts_jobs,

            "batch_size": BATCH_SIZE,
            "batch_max_wait_ms": BATCH_MAX_WAIT_MS,
            "busy_backlog_chunks": BUSY_BACKLOG_CHUNKS,
            "voice_clone_prompt_cache_size": len(voice_prompt_cache),
            "voice_clone_prompt_cache_max_size": PROMPT_CACHE_SIZE,
            "tts_jobs": job_counts,
        }


async def _try_reserve_generation_chunks(count: int) -> tuple[bool, int]:
    global queued_generations
    async with metrics_lock:
        outstanding_chunks = queued_generations + active_generations
        if outstanding_chunks >= BUSY_BACKLOG_CHUNKS:
            return False, outstanding_chunks
        queued_generations += count
        return True, outstanding_chunks


def _batch_generation_key(
    item: ChunkGenerationItem,
) -> tuple[int, Optional[float], Optional[float], Optional[str]]:
    return (
        item.req.num_step,
        item.req.speed,
        item.req.guidance_scale,
        item.req.language,
    )


def _group_batch_items(items: list[ChunkGenerationItem]) -> list[list[ChunkGenerationItem]]:
    groups: dict[
        tuple[int, Optional[float], Optional[float], Optional[str]],
        list[ChunkGenerationItem],
    ] = {}
    for item in items:
        groups.setdefault(_batch_generation_key(item), []).append(item)
    return list(groups.values())


def _take_deferred_generation_item(
    key: Optional[tuple[int, Optional[float], Optional[float], Optional[str]]] = None,
) -> Optional[ChunkGenerationItem]:
    if not deferred_generation_items:
        return None
    for index, item in enumerate(deferred_generation_items):
        if key is not None and _batch_generation_key(item) != key:
            continue
        del deferred_generation_items[index]
        return item
    return None


async def _collect_generation_batch(first_item: ChunkGenerationItem) -> list[ChunkGenerationItem]:
    batch = [first_item]
    batch_key = _batch_generation_key(first_item)
    deadline = asyncio.get_running_loop().time() + (BATCH_MAX_WAIT_MS / 1000)

    # Phase 1: Drain deferred items with matching batch key (instant, no waiting)
    while len(batch) < BATCH_SIZE:
        deferred_match = _take_deferred_generation_item(batch_key)
        if deferred_match is None:
            break
        batch.append(deferred_match)

    # Phase 2: Wait for new items from queue up to deadline
    while len(batch) < BATCH_SIZE:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        try:
            candidate = await asyncio.wait_for(generation_queue.get(), timeout=remaining)
            if _batch_generation_key(candidate) == batch_key:
                batch.append(candidate)
            else:
                deferred_generation_items.append(candidate)
        except TimeoutError:
            break

    logger.info(
        "Batch collected: items=%d/%d key=%s deferred_remaining=%d",
        len(batch),
        BATCH_SIZE,
        batch_key,
        len(deferred_generation_items),
    )
    return batch


async def _increment_job_chunks(request_id: str, *, completed: int = 0, failed: int = 0) -> None:
    async with tts_jobs_lock:
        job = tts_jobs.get(request_id)
        if job is None:
            return
        job.chunks_completed += completed
        job.chunks_failed += failed
        job.updated_at = time.time()


async def _fail_unfinished_job_chunks(request_id: str) -> int:
    async with tts_jobs_lock:
        job = tts_jobs.get(request_id)
        if job is None:
            return 0
        unfinished = max(job.chunks_total - job.chunks_completed - job.chunks_failed, 0)
        if unfinished:
            job.chunks_failed += unfinished
            job.updated_at = time.time()
        return unfinished


async def _run_generation_batch(items: list[ChunkGenerationItem]) -> None:
    live_items = [item for item in items if not item.future.cancelled()]
    await _queued_generation_delta(-len(items))
    if not live_items:
        for item in items:
            generation_queue.task_done()
        return

    await _active_generation_batch_delta(1)
    await _active_generation_delta(len(live_items))
    try:
        if SKIP_MODEL_LOAD:
            for item in live_items:
                _write_silent_test_wav(item.output_wav)
            transcripts = ["" for _ in live_items]
        else:
            async with gpu_generation_semaphore:
                transcripts = await _run_gpu_operation_with_oom_retry(
                    "TTS batch",
                    lambda: asyncio.to_thread(_generate_wav_batch, live_items),
                )
        if len(transcripts) != len(live_items):
            raise RuntimeError(
                f"Batch generated {len(transcripts)} outputs for {len(live_items)} requests."
            )
        for item, transcript in zip(live_items, transcripts):
            if not item.future.done():
                item.future.set_result(transcript)
                await _increment_job_chunks(item.request_id, completed=1)
            elif item.future.cancelled():
                _remove_files([item.output_wav])
    except Exception as exc:
        for item in live_items:
            if not item.future.done():
                item.future.set_exception(exc)
                await _increment_job_chunks(item.request_id, failed=1)
    finally:
        await _active_generation_delta(-len(live_items))
        await _active_generation_batch_delta(-1)
        for item in items:
            generation_queue.task_done()


async def _generation_batch_worker() -> None:
    while True:
        first_item = _take_deferred_generation_item()
        if first_item is None:
            first_item = await generation_queue.get()
        batch = await _collect_generation_batch(first_item)

        for group in _group_batch_items(batch):
            await _run_generation_batch(group)


async def _generate_per_chunk_audio(
    request_id: str,
    req: TTSRequest,
    ref: ReferenceCacheEntry,
    chunks: list[str],
    cleanup_paths: list[Path],
) -> tuple[list[Path], str, str]:
    transcript = ref.transcript or ""
    if SKIP_MODEL_LOAD:
        voice_clone_prompt = None
    else:
        voice_clone_prompt = await _get_voice_clone_prompt(req, ref)
        transcript = (voice_clone_prompt.ref_text or "").strip()

    futures: list[asyncio.Future[str]] = []
    chunk_wav_paths: list[Path] = []
    for chunk_index, chunk_text in enumerate(chunks):
        chunk_wav = Path(
            tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TMP_DIR).name
        )
        cleanup_paths.append(chunk_wav)
        chunk_wav_paths.append(chunk_wav)
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        item = ChunkGenerationItem(
            request_id=request_id,
            chunk_index=chunk_index,
            text=chunk_text,
            req=req,
            voice_clone_prompt=voice_clone_prompt,
            output_wav=chunk_wav,
            future=future,
        )
        await generation_queue.put(item)
        futures.append(future)

    results = await asyncio.gather(*futures, return_exceptions=True)
    errors = [result for result in results if isinstance(result, Exception)]
    if errors:
        raise RuntimeError(f"{len(errors)} TTS chunk(s) failed; first error: {errors[0]!r}")

    for result in results:
        if isinstance(result, str) and result.strip():
            transcript = result.strip()
            break

    if req.format == "mp3":
        chunk_output_paths: list[Path] = []
        for chunk_wav in chunk_wav_paths:
            chunk_mp3 = Path(
                tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=TMP_DIR).name
            )
            cleanup_paths.append(chunk_mp3)
            await asyncio.to_thread(_convert_wav_to_mp3, chunk_wav, chunk_mp3)
            chunk_output_paths.append(chunk_mp3)
        _remove_files(chunk_wav_paths)
        wav_set = set(chunk_wav_paths)
        cleanup_paths[:] = [p for p in cleanup_paths if p not in wav_set]
        return chunk_output_paths, "audio/mpeg", transcript

    return chunk_wav_paths, "audio/wav", transcript


def _remove_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove temp file %s: %s", path, exc)


async def _set_job_state(request_id: str, **updates: object) -> None:
    async with tts_jobs_lock:
        job = tts_jobs.get(request_id)
        if job is None:
            return
        for key, value in updates.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def _job_payload(job: TTSJob) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": job.request_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "status_url": f"/v1/tts/jobs/{job.request_id}",
        "language": job.language,
        "input_chars": job.input_chars,
        "chunks_total": job.chunks_total,
        "chunks_completed": job.chunks_completed,
        "chunks_failed": job.chunks_failed,
    }
    if job.detail:
        payload["detail"] = job.detail
    if job.audio_cache_hit is not None:
        payload["cache_hit"] = job.audio_cache_hit
    if job.status == "succeeded":
        payload.update(
            {
                "format": job.format,
                "audio_url": f"/v1/tts/jobs/{job.request_id}/audio",
                "transcript": job.transcript,
            }
        )
    return payload


async def _cleanup_expired_jobs() -> None:
    if JOB_TTL_SECONDS <= 0:
        return
    now = time.time()
    expired: list[TTSJob] = []
    async with tts_jobs_lock:
        for request_id, job in list(tts_jobs.items()):
            if job.status in {"queued", "running"}:
                continue
            if now - job.updated_at <= JOB_TTL_SECONDS:
                continue
            expired.append(job)
            del tts_jobs[request_id]

    for job in expired:
        if job.cleanup_paths:
            _remove_files(job.cleanup_paths)
    if expired:
        logger.info("Cleaned up %d expired TTS jobs.", len(expired))


async def _periodic_job_cleanup() -> None:
    while True:
        await asyncio.sleep(JOB_CLEANUP_INTERVAL_SECONDS)
        try:
            await _cleanup_expired_jobs()
        except Exception:
            logger.exception("Error during periodic job cleanup")


async def _run_tts_job(request_id: str, req: TTSRequest) -> None:
    await _set_job_state(request_id, status="running", detail=None)
    cleanup_paths: list[Path] = []

    try:
        ref = await _resolve_reference(req)
        chunk_output_paths, media_type, transcript = await _generate_per_chunk_audio(
            request_id,
            req,
            ref,
            req.chunks,
            cleanup_paths,
        )

        ext = "mp3" if req.format == "mp3" else "wav"
        final_chunk_paths: list[Path] = []
        for i, src_path in enumerate(chunk_output_paths):
            dst_path = JOB_DIR / f"{request_id}_chunk{i}.{ext}"
            src_path.replace(dst_path)
            cleanup_paths = [p for p in cleanup_paths if p != src_path]
            cleanup_paths.append(dst_path)
            final_chunk_paths.append(dst_path)

        await _set_job_state(
            request_id,
            status="succeeded",
            detail=None,
            chunk_paths=final_chunk_paths,
            chunk_media_type=media_type,
            transcript=transcript,
            audio_cache_hit=ref.audio_cache_hit,
            cleanup_paths=cleanup_paths,
        )
        logger.info("TTS async job %s succeeded (%d chunks)", request_id, len(final_chunk_paths))
    except Exception as exc:
        unfinished = await _fail_unfinished_job_chunks(request_id)
        if unfinished:
            await _queued_generation_delta(-unfinished)
        _remove_files(cleanup_paths)
        await _set_job_state(
            request_id,
            status="failed",
            detail=f"TTS generation failed: {exc}",
            cleanup_paths=[],
        )
        logger.exception("TTS async job %s failed", request_id)


def _validate_token(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="API_TOKEN is not configured.")
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _validate_request(req: TTSRequest) -> None:
    req.format = req.format.lower().strip()
    if req.format not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="format must be wav or mp3.")
    if not req.chunks or not any(c.strip() for c in req.chunks):
        raise HTTPException(status_code=400, detail="chunks is required and must contain non-empty strings.")
    req.chunks = [c.strip() for c in req.chunks if c.strip()]
    if not req.ref_text or not req.ref_text.strip():
        raise HTTPException(status_code=400, detail="ref_text is required.")
    if not req.ref_audio_url or not req.ref_audio_url.strip():
        raise HTTPException(status_code=400, detail="ref_audio_url is required.")
    req.language = _normalize_language(req.language)
    parsed = urlparse(req.ref_audio_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="ref_audio_url must be http(s).")
    if req.num_step < 4 or req.num_step > 64:
        raise HTTPException(status_code=400, detail="num_step must be between 4 and 64.")
    if req.speed is not None and (req.speed < 0.5 or req.speed > 2.0):
        raise HTTPException(status_code=400, detail="speed must be between 0.5 and 2.0.")
    if req.guidance_scale is not None and (req.guidance_scale < 0.0 or req.guidance_scale > 4.0):
        raise HTTPException(status_code=400, detail="guidance_scale must be between 0.0 and 4.0.")


@app.on_event("startup")
async def startup() -> None:
    global generation_batch_worker_task, job_cleanup_task
    _ensure_dirs()
    await asyncio.to_thread(_load_model)
    generation_batch_worker_task = asyncio.create_task(_generation_batch_worker())
    logger.info(
        "OmniVoice chunk scheduler started: batch_size=%s max_wait_ms=%s busy_backlog_chunks=%s.",
        BATCH_SIZE,
        BATCH_MAX_WAIT_MS,
        BUSY_BACKLOG_CHUNKS,
    )
    job_cleanup_task = asyncio.create_task(_periodic_job_cleanup())
    logger.info("Periodic job cleanup started (interval=%ds).", JOB_CLEANUP_INTERVAL_SECONDS)

@app.on_event("shutdown")
async def shutdown() -> None:
    if generation_batch_worker_task is not None:
        generation_batch_worker_task.cancel()
        try:
            await generation_batch_worker_task
        except asyncio.CancelledError:
            pass


@app.get("/health")
async def health() -> dict[str, object]:
    gpu = resolved_gpu_name
    try:
        import torch

        if gpu is None and torch.cuda.is_available():
            gpu_index = _cuda_device_index()
            if gpu_index < torch.cuda.device_count():
                gpu = torch.cuda.get_device_name(gpu_index)
    except Exception:
        gpu = resolved_gpu_name

    runtime_metrics = await _get_runtime_metrics()
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model": MODEL_NAME,
        "gpu": gpu,
        "gpu_profile": resolved_gpu_profile or _gpu_profile_from_name(gpu),
        "requested_gpu_profile": GPU_PROFILE,
        "dtype": resolved_model_dtype or MODEL_DTYPE,
        "cache_audio_count": len(list(REF_AUDIO_DIR.glob("*"))),
        "cache_transcript_count": len(list(TRANSCRIPT_DIR.glob("*.json"))),
        "acceleration": ACCELERATION,
        **runtime_metrics,
    }


@app.post("/v1/cache/clear")
async def clear_cache(_: None = Depends(_validate_token)) -> dict:
    import shutil
    try:
        if REF_AUDIO_DIR.exists():
            shutil.rmtree(REF_AUDIO_DIR)
            REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        if TRANSCRIPT_DIR.exists():
            shutil.rmtree(TRANSCRIPT_DIR)
            TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        async with voice_prompt_cache_guard:
            voice_prompt_cache.clear()
        logger.info("Cache cleared successfully via API.")
        return {"status": "ok", "message": "All cached reference audios, transcripts, and memory prompts cleared successfully."}
    except Exception as e:
        logger.exception("Failed to clear cache.")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")


@app.post("/v1/tts")
async def tts(
    req: TTSRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(_validate_token),
) -> JSONResponse:
    _validate_request(req)
    chunks = req.chunks

    reserved, outstanding_chunks = await _try_reserve_generation_chunks(len(chunks))
    if not reserved:
        raise HTTPException(
            status_code=429,
            detail=(
                "TTS chunk backlog is busy; retry later. "
                f"outstanding_chunks={outstanding_chunks}, requested_chunks={len(chunks)}, "
                f"limit={BUSY_BACKLOG_CHUNKS}."
            ),
            headers={
                "Retry-After": "1",
                "X-Busy-Backlog-Chunks": str(BUSY_BACKLOG_CHUNKS),
                "X-Outstanding-Chunks": str(outstanding_chunks),
                "X-Requested-Chunks": str(len(chunks)),
            },
        )

    request_id = str(uuid.uuid4())
    now = time.time()
    job = TTSJob(
        request_id=request_id,
        status="queued",
        created_at=now,
        updated_at=now,
        format=req.format,
        language=req.language,
        chunks_total=len(chunks),
        input_chars=sum(len(c) for c in chunks),
    )
    async with tts_jobs_lock:
        tts_jobs[request_id] = job

    background_tasks.add_task(_run_tts_job, request_id, req)
    return JSONResponse(
        status_code=202,
        content=_job_payload(job),
        headers={
            "X-Request-Id": request_id,
            "Location": f"/v1/tts/jobs/{request_id}",
        },
    )


@app.get("/v1/tts/jobs/{request_id}")
async def get_tts_job(
    request_id: str,
    _: None = Depends(_validate_token),
) -> JSONResponse:
    async with tts_jobs_lock:
        job = tts_jobs.get(request_id)
        if job is None:
            raise HTTPException(status_code=404, detail="TTS job not found.")
        payload = _job_payload(job)
    return JSONResponse(content=payload)


@app.get("/v1/tts/jobs/{request_id}/audio")
async def get_tts_job_audio(
    request_id: str,
    _: None = Depends(_validate_token),
) -> Response:
    async with tts_jobs_lock:
        job = tts_jobs.get(request_id)
        if job is None:
            raise HTTPException(status_code=404, detail="TTS job not found.")
        if job.status != "succeeded":
            raise HTTPException(status_code=409, detail=f"TTS job is {job.status}.")
        if not job.chunk_paths:
            raise HTTPException(status_code=500, detail="TTS job audio is missing.")
        chunk_paths = list(job.chunk_paths)
        media_type = job.chunk_media_type or "audio/wav"
        transcript = job.transcript
        cache_hit = job.audio_cache_hit

    for path in chunk_paths:
        if not path.exists():
            raise HTTPException(status_code=410, detail="TTS job audio expired.")

    async def stream_length_prefixed():
        yield len(chunk_paths).to_bytes(4, "big")
        for path in chunk_paths:
            yield path.stat().st_size.to_bytes(4, "big")
            with path.open("rb") as handle:
                while True:
                    block = handle.read(STREAM_CHUNK_SIZE_BYTES)
                    if not block:
                        break
                    yield block

    headers = {
        "X-Request-Id": request_id,
        "X-Cache-Hit": str(bool(cache_hit)).lower(),
        "X-Transcript": quote(transcript or "", safe=""),
        "X-Transcript-Encoding": "urlencoded-utf8",
        "X-Chunks-Total": str(len(chunk_paths)),
        "X-Audio-Format": media_type,
    }
    return StreamingResponse(
        stream_length_prefixed(),
        media_type="application/octet-stream",
        headers=headers,
    )
