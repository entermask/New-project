import logging
import os
from typing import Optional


logger = logging.getLogger("omnivoice-api.cuda")


DEVICE_MAP = os.getenv("OMNIVOICE_DEVICE", os.getenv("QWEN_ASR_DEVICE", "cuda:0"))


def cuda_device_index(device: Optional[str] = None) -> int:
    value = device or DEVICE_MAP
    if value.startswith("cuda:"):
        try:
            return int(value.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def is_cuda_oom_error(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass

    message = str(exc).lower()
    return (
        "cuda out of memory" in message
        or "cublas_status_alloc_failed" in message
        or "cuda error: out of memory" in message
    )


def clear_cuda_cache(context: str, device: Optional[str] = None) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        index = cuda_device_index(device)
        with torch.cuda.device(index):
            free_before, total = torch.cuda.mem_get_info()
            torch.cuda.empty_cache()
            free_after, _ = torch.cuda.mem_get_info()
        logger.warning(
            "%s: cleared CUDA cache on device %s free_before=%.2fGiB free_after=%.2fGiB total=%.2fGiB",
            context,
            index,
            free_before / 1024**3,
            free_after / 1024**3,
            total / 1024**3,
        )
    except Exception as exc:
        logger.warning("%s: could not clear CUDA cache: %s", context, exc)


def gpu_snapshot(device: Optional[str] = None) -> dict[str, object]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        index = cuda_device_index(device)
        name = torch.cuda.get_device_name(index)
        free, total = torch.cuda.mem_get_info(index)
        return {
            "available": True,
            "index": index,
            "name": name,
            "memory_free_bytes": free,
            "memory_total_bytes": total,
            "memory_used_bytes": total - free,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
