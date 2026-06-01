import asyncio
import logging
import os
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import soundfile as sf
from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import JSONResponse

from omnivoice_api.common.auth import validate_token
from omnivoice_api.common.cuda import clear_cuda_cache
from omnivoice_api.common.cuda import gpu_snapshot
from omnivoice_api.common.cuda import is_cuda_oom_error


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("qwen3-asr-api")


BACKEND = os.getenv("QWEN_ASR_BACKEND", "vllm").lower().strip()
DEFAULT_MODEL = os.getenv("QWEN_ASR_DEFAULT_MODEL", "0.6b").lower().strip()
MODEL_IDS = {
    "0.6b": os.getenv("QWEN_ASR_06B_MODEL", "Qwen/Qwen3-ASR-0.6B"),
    "1.7b": os.getenv("QWEN_ASR_17B_MODEL", "Qwen/Qwen3-ASR-1.7B"),
}
MODEL_ALIASES = {
    "0.6b": "0.6b",
    "06b": "0.6b",
    "0_6b": "0.6b",
    "600m": "0.6b",
    "qwen3-asr-0.6b": "0.6b",
    "1.7b": "1.7b",
    "17b": "1.7b",
    "1_7b": "1.7b",
    "qwen3-asr-1.7b": "1.7b",
}
LANGUAGE_ALIASES = {
    "en": "English",
    "eng": "English",
    "english": "English",
    "vi": "Vietnamese",
    "vie": "Vietnamese",
    "vietnamese": "Vietnamese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "cmn": "Chinese",
    "chinese": "Chinese",
    "yue": "Cantonese",
    "zh-hk": "Cantonese",
    "cantonese": "Cantonese",
    "ar": "Arabic",
    "ara": "Arabic",
    "arabic": "Arabic",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "es": "Spanish",
    "spanish": "Spanish",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "id": "Indonesian",
    "indonesian": "Indonesian",
    "it": "Italian",
    "italian": "Italian",
    "ko": "Korean",
    "korean": "Korean",
    "ru": "Russian",
    "russian": "Russian",
    "th": "Thai",
    "thai": "Thai",
    "ja": "Japanese",
    "japanese": "Japanese",
    "tr": "Turkish",
    "turkish": "Turkish",
    "hi": "Hindi",
    "hindi": "Hindi",
    "ms": "Malay",
    "malay": "Malay",
    "nl": "Dutch",
    "dutch": "Dutch",
    "sv": "Swedish",
    "swedish": "Swedish",
    "da": "Danish",
    "danish": "Danish",
    "fi": "Finnish",
    "finnish": "Finnish",
    "pl": "Polish",
    "polish": "Polish",
    "cs": "Czech",
    "czech": "Czech",
    "fil": "Filipino",
    "tl": "Filipino",
    "filipino": "Filipino",
    "fa": "Persian",
    "persian": "Persian",
    "el": "Greek",
    "greek": "Greek",
    "ro": "Romanian",
    "romanian": "Romanian",
    "hu": "Hungarian",
    "hungarian": "Hungarian",
    "mk": "Macedonian",
    "macedonian": "Macedonian",
}
DEVICE_MAP = os.getenv("QWEN_ASR_DEVICE", "cuda:0")
DTYPE = os.getenv("QWEN_ASR_DTYPE", "bfloat16").lower().strip()
GPU_MEMORY_UTILIZATION = float(os.getenv("QWEN_ASR_GPU_MEMORY_UTILIZATION", "0.35"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_ASR_MAX_NEW_TOKENS", "4096"))
VLLM_MAX_MODEL_LEN = int(os.getenv("QWEN_ASR_VLLM_MAX_MODEL_LEN", "8192"))
VLLM_ATTENTION_BACKEND = os.getenv("QWEN_ASR_VLLM_ATTENTION_BACKEND", "TRITON_ATTN").strip()
VLLM_MM_ENCODER_ATTN_BACKEND = os.getenv("QWEN_ASR_VLLM_MM_ENCODER_ATTN_BACKEND", "TORCH_SDPA").strip()
VLLM_LIMIT_AUDIO_PER_PROMPT = int(os.getenv("QWEN_ASR_VLLM_LIMIT_AUDIO_PER_PROMPT", "1"))
VLLM_ENFORCE_EAGER = os.getenv("QWEN_ASR_VLLM_ENFORCE_EAGER", "0") == "1"
INITIAL_BATCH_SIZE = max(1, int(os.getenv("QWEN_ASR_MAX_INFERENCE_BATCH_SIZE", "32")))
BATCH_MAX_WAIT_MS = max(0.0, float(os.getenv("QWEN_ASR_BATCH_MAX_WAIT_MS", "50")))
MAX_QUEUE_ITEMS = max(1, int(os.getenv("QWEN_ASR_MAX_QUEUE_ITEMS", "512")))
REQUEST_TIMEOUT = float(os.getenv("QWEN_ASR_REQUEST_TIMEOUT", "600"))
JOB_TTL_SECONDS = int(os.getenv("QWEN_ASR_JOB_TTL_SECONDS", "3600"))
TMP_DIR = Path(os.getenv("QWEN_ASR_TMP_DIR", "/ephemeral/qwen-asr/tmp"))
SKIP_MODEL_LOAD = os.getenv("QWEN_ASR_SKIP_MODEL_LOAD", "0") == "1"
ENABLE_FORCED_ALIGNER = os.getenv("QWEN_ASR_ENABLE_FORCED_ALIGNER", "0") == "1"
FORCED_ALIGNER_MODEL = os.getenv("QWEN_ASR_FORCED_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
DEFAULT_CONTEXT = os.getenv("QWEN_ASR_DEFAULT_CONTEXT", "").strip()
JOB_CLEANUP_INTERVAL_SECONDS = 60


app = FastAPI(title="Qwen3 ASR API", version="1.0.0")
stt_jobs: dict[str, "STTJob"] = {}
stt_jobs_lock = asyncio.Lock()
model_states: dict[str, "ModelState"] = {}
worker_tasks: list[asyncio.Task] = []
cleanup_task: Optional[asyncio.Task] = None


@dataclass
class STTJob:
    job_id: str
    status: str
    created_at: float
    updated_at: float
    model: str
    context: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    file_path: Optional[Path] = None


@dataclass
class STTWorkItem:
    job_id: str
    file_path: Path
    model_key: str
    language: Optional[str]
    context: Optional[str]
    word_timestamps: bool
    future: asyncio.Future
    cleanup_after_done: bool = False


@dataclass
class ModelState:
    key: str
    repo_id: str
    model: object = None
    loaded: bool = False
    load_error: Optional[str] = None
    current_batch_size: int = INITIAL_BATCH_SIZE
    oom_count: int = 0
    retry_count: int = 0
    completed_items: int = 0
    failed_items: int = 0
    queue: asyncio.Queue[STTWorkItem] = None


def _normalize_model_key(value: Optional[str]) -> str:
    raw = (value or DEFAULT_MODEL or "0.6b").strip().lower()
    raw = raw.replace(" ", "")
    key = MODEL_ALIASES.get(raw, raw)
    if key not in MODEL_IDS:
        raise HTTPException(status_code=400, detail="model must be one of: 0.6b, 1.7b.")
    return key


def _normalize_language(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    raw = value.strip()
    if not raw or raw.lower() in {"auto", "detect", "none", "null"}:
        return None
    return LANGUAGE_ALIASES.get(raw.lower().replace("_", "-"), raw)


def _torch_dtype():
    import torch

    if DTYPE in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if DTYPE in {"fp16", "float16", "half"}:
        return torch.float16
    if DTYPE in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("QWEN_ASR_DTYPE must be bfloat16, float16, or float32.")


def _load_qwen_model(repo_id: str) -> object:
    from qwen_asr import Qwen3ASRModel

    forced_aligner = FORCED_ALIGNER_MODEL if ENABLE_FORCED_ALIGNER else None
    if BACKEND == "vllm":
        vllm_kwargs = {
            "forced_aligner": forced_aligner,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_inference_batch_size": INITIAL_BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "dtype": DTYPE,
            "max_model_len": VLLM_MAX_MODEL_LEN,
            "mm_encoder_attn_backend": VLLM_MM_ENCODER_ATTN_BACKEND,
            "limit_mm_per_prompt": {"audio": VLLM_LIMIT_AUDIO_PER_PROMPT},
            "enforce_eager": VLLM_ENFORCE_EAGER,
        }
        if VLLM_ATTENTION_BACKEND:
            vllm_kwargs["attention_config"] = {"backend": VLLM_ATTENTION_BACKEND}
        return Qwen3ASRModel.LLM(
            repo_id,
            **vllm_kwargs,
        )
    if BACKEND == "transformers":
        return Qwen3ASRModel.from_pretrained(
            repo_id,
            forced_aligner=forced_aligner,
            device_map=DEVICE_MAP,
            torch_dtype=_torch_dtype(),
            attn_implementation=os.getenv("QWEN_ASR_ATTENTION", "sdpa"),
            max_inference_batch_size=INITIAL_BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    raise ValueError("QWEN_ASR_BACKEND must be vllm or transformers.")


def _load_models() -> None:
    for key, repo_id in MODEL_IDS.items():
        state = model_states[key]
        if SKIP_MODEL_LOAD:
            logger.warning("QWEN_ASR_SKIP_MODEL_LOAD=1; not loading %s.", repo_id)
            state.loaded = False
            continue
        logger.info(
            "Loading Qwen3-ASR model key=%s repo=%s backend=%s batch=%d max_new_tokens=%d ...",
            key,
            repo_id,
            BACKEND,
            state.current_batch_size,
            MAX_NEW_TOKENS,
        )
        try:
            state.model = _load_qwen_model(repo_id)
            state.loaded = True
            state.load_error = None
            logger.info("Loaded Qwen3-ASR model key=%s repo=%s.", key, repo_id)
        except Exception as exc:
            state.loaded = False
            state.load_error = str(exc)
            logger.exception("Failed to load Qwen3-ASR model key=%s repo=%s.", key, repo_id)
            raise


def _format_srt_timestamp(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _generate_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(f"{_format_srt_timestamp(segment['start'])} --> {_format_srt_timestamp(segment['end'])}")
        lines.append(str(segment.get("text", "")).strip())
        lines.append("")
    return "\n".join(lines)


def _audio_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    return 0.0


def _result_to_payload(result: object, file_path: Path, requested_model: str) -> dict[str, object]:
    text = (getattr(result, "text", None) or getattr(result, "transcription", None) or str(result)).strip()
    language = getattr(result, "language", None)
    timestamps = (
        getattr(result, "time_stamps", None)
        or getattr(result, "timestamps", None)
        or getattr(result, "segments", None)
    )
    duration = _audio_duration(file_path)

    segments: list[dict[str, object]] = []
    if isinstance(timestamps, list) and timestamps:
        for item in timestamps:
            start = float(getattr(item, "start", item.get("start", 0.0) if isinstance(item, dict) else 0.0))
            end = float(getattr(item, "end", item.get("end", start) if isinstance(item, dict) else start))
            seg_text = getattr(item, "text", item.get("text", "") if isinstance(item, dict) else "")
            segments.append({"start": start, "end": end, "text": seg_text})
    else:
        segments.append({"start": 0.0, "end": duration, "text": text})

    return {
        "status": "completed",
        "model": requested_model,
        "backend": BACKEND,
        "language": language,
        "language_probability": None,
        "duration_seconds": duration,
        "text": text,
        "segments": segments,
        "srt": _generate_srt(segments),
    }


async def _run_model_batch(state: ModelState, batch: list[STTWorkItem]) -> list[dict[str, object]]:
    if SKIP_MODEL_LOAD:
        return [
            {
                "status": "completed",
                "model": item.model_key,
                "backend": BACKEND,
                "language": item.language,
                "language_probability": None,
                "duration_seconds": _audio_duration(item.file_path),
                "text": "",
                "segments": [{"start": 0.0, "end": _audio_duration(item.file_path), "text": ""}],
                "srt": _generate_srt([{"start": 0.0, "end": _audio_duration(item.file_path), "text": ""}]),
            }
            for item in batch
        ]

    if not state.loaded or state.model is None:
        raise RuntimeError(f"Qwen3-ASR model {state.key} is not loaded: {state.load_error or 'unknown error'}")

    def transcribe() -> list[dict[str, object]]:
        outputs = state.model.transcribe(
            audio=[str(item.file_path) for item in batch],
            context=[item.context or DEFAULT_CONTEXT for item in batch],
            language=[_normalize_language(item.language) for item in batch],
            return_time_stamps=any(item.word_timestamps for item in batch),
        )
        return [
            _result_to_payload(result, item.file_path, item.model_key)
            for result, item in zip(outputs, batch)
        ]

    return await asyncio.to_thread(transcribe)


async def _run_batch_with_oom_retry(state: ModelState, batch: list[STTWorkItem]) -> list[dict[str, object]]:
    try:
        return await _run_model_batch(state, batch)
    except Exception as exc:
        if not is_cuda_oom_error(exc):
            raise

        state.oom_count += 1
        state.retry_count += 1
        state.current_batch_size = max(1, state.current_batch_size // 2)
        clear_cuda_cache(f"Qwen3-ASR {state.key} OOM")
        logger.warning(
            "Qwen3-ASR OOM key=%s batch=%d; reduced dynamic batch limit to %d and retrying once.",
            state.key,
            len(batch),
            state.current_batch_size,
        )
        try:
            return await _run_model_batch(state, batch)
        except Exception as retry_exc:
            if not is_cuda_oom_error(retry_exc) or len(batch) == 1:
                raise

            state.oom_count += 1
            clear_cuda_cache(f"Qwen3-ASR {state.key} OOM split")
            results: list[dict[str, object]] = []
            for item in batch:
                try:
                    results.extend(await _run_batch_with_oom_retry(state, [item]))
                except Exception:
                    raise
            return results


async def _model_worker(state: ModelState) -> None:
    while True:
        first = await state.queue.get()
        batch = [first]
        deadline = time.monotonic() + (BATCH_MAX_WAIT_MS / 1000.0)
        while len(batch) < state.current_batch_size:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(state.queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            batch.append(item)

        try:
            results = await _run_batch_with_oom_retry(state, batch)
            for item, result in zip(batch, results):
                state.completed_items += 1
                await _complete_item(item, result=result)
        except Exception as exc:
            logger.exception("Qwen3-ASR batch failed key=%s size=%d.", state.key, len(batch))
            for item in batch:
                state.failed_items += 1
                await _complete_item(item, error=str(exc))
        finally:
            for _ in batch:
                state.queue.task_done()


async def _complete_item(item: STTWorkItem, result: Optional[dict] = None, error: Optional[str] = None) -> None:
    if result is not None:
        async with stt_jobs_lock:
            job = stt_jobs.get(item.job_id)
            if job:
                job.status = "completed"
                job.result = result
                job.updated_at = time.time()
        if not item.future.done():
            item.future.set_result(result)
    else:
        async with stt_jobs_lock:
            job = stt_jobs.get(item.job_id)
            if job:
                job.status = "failed"
                job.error = error or "STT failed."
                job.updated_at = time.time()
        if not item.future.done():
            item.future.set_exception(RuntimeError(error or "STT failed."))

    if item.cleanup_after_done and item.file_path.exists():
        try:
            item.file_path.unlink()
        except OSError:
            pass


async def _cleanup_expired_jobs() -> None:
    now = time.time()
    expired: list[STTJob] = []
    async with stt_jobs_lock:
        for job_id, job in list(stt_jobs.items()):
            if now - job.updated_at > JOB_TTL_SECONDS:
                expired.append(job)
                del stt_jobs[job_id]
    for job in expired:
        if job.file_path and job.file_path.exists():
            try:
                job.file_path.unlink()
            except OSError:
                pass
    if expired:
        logger.info("Cleaned up %d expired STT jobs.", len(expired))


async def _periodic_cleanup() -> None:
    while True:
        await asyncio.sleep(JOB_CLEANUP_INTERVAL_SECONDS)
        await _cleanup_expired_jobs()


def _ensure_dirs() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def _queue_size() -> int:
    return sum(state.queue.qsize() for state in model_states.values())


def _save_upload(file: UploadFile, job_id: str) -> Path:
    suffix = Path(file.filename or ".wav").suffix or ".wav"
    path = TMP_DIR / f"{job_id}{suffix}"
    try:
        with path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}") from exc
    return path


async def _enqueue(item: STTWorkItem) -> None:
    if _queue_size() >= MAX_QUEUE_ITEMS:
        raise HTTPException(
            status_code=429,
            detail=f"STT queue is busy; queued_items={_queue_size()}, limit={MAX_QUEUE_ITEMS}.",
            headers={"Retry-After": "1"},
        )
    await model_states[item.model_key].queue.put(item)


@app.on_event("startup")
async def startup() -> None:
    global cleanup_task
    _ensure_dirs()
    for key, repo_id in MODEL_IDS.items():
        model_states[key] = ModelState(key=key, repo_id=repo_id, queue=asyncio.Queue())
    await asyncio.to_thread(_load_models)
    for state in model_states.values():
        worker_tasks.append(asyncio.create_task(_model_worker(state)))
    cleanup_task = asyncio.create_task(_periodic_cleanup())


@app.on_event("shutdown")
async def shutdown() -> None:
    for task in worker_tasks:
        task.cancel()
    if cleanup_task is not None:
        cleanup_task.cancel()


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "backend": BACKEND,
        "default_model": DEFAULT_MODEL,
        "models": {
            key: {
                "repo_id": state.repo_id,
                "loaded": state.loaded,
                "load_error": state.load_error,
                "current_batch_size": state.current_batch_size,
                "queue_size": state.queue.qsize(),
                "oom_count": state.oom_count,
                "retry_count": state.retry_count,
                "completed_items": state.completed_items,
                "failed_items": state.failed_items,
            }
            for key, state in model_states.items()
        },
        "gpu": gpu_snapshot(DEVICE_MAP),
    }


@app.post("/v1/stt/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    mode: str = Form("sync"),
    beam_size: int = Form(5),
    vad_filter: bool = Form(True),
    word_timestamps: bool = Form(False),
    model: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    _: None = Depends(validate_token),
) -> JSONResponse:
    del background_tasks, beam_size, vad_filter
    if mode not in {"sync", "async"}:
        raise HTTPException(status_code=400, detail="mode must be 'sync' or 'async'.")

    model_key = _normalize_model_key(model)
    job_id = f"stt_job_{uuid.uuid4().hex[:12]}"
    temp_file_path = _save_upload(file, job_id)
    now = time.time()
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    cleanup_after_done = mode == "sync"
    job = STTJob(
        job_id=job_id,
        status="queued",
        created_at=now,
        updated_at=now,
        model=model_key,
        context=(context or DEFAULT_CONTEXT or None),
        file_path=temp_file_path,
    )
    async with stt_jobs_lock:
        stt_jobs[job_id] = job

    item = STTWorkItem(
        job_id=job_id,
        file_path=temp_file_path,
        model_key=model_key,
        language=language or None,
        context=(context or DEFAULT_CONTEXT or None),
        word_timestamps=word_timestamps,
        future=future,
        cleanup_after_done=cleanup_after_done,
    )
    try:
        await _enqueue(item)
    except Exception:
        async with stt_jobs_lock:
            stt_jobs.pop(job_id, None)
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except OSError:
                pass
        raise

    if mode == "async":
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "queued",
                "created_at": now,
                "model": model_key,
                "context_applied": bool(context or DEFAULT_CONTEXT),
                "status_url": f"/v1/stt/status/{job_id}",
            },
        )

    try:
        result = await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT)
        async with stt_jobs_lock:
            stt_jobs.pop(job_id, None)
        return JSONResponse(content=result)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="STT transcription timed out.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/stt/status/{job_id}")
async def get_stt_job_status(
    job_id: str,
    _: None = Depends(validate_token),
) -> JSONResponse:
    async with stt_jobs_lock:
        job = stt_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="STT job not found.")
        payload: dict[str, object] = {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "model": job.model,
            "context_applied": bool(job.context),
        }
        if job.status == "completed":
            payload["result"] = job.result
        elif job.status == "failed":
            payload["error"] = job.error
    return JSONResponse(content=payload)
