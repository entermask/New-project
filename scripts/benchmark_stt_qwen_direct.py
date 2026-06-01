#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def make_audio(seconds: float, sr: int, index: int) -> tuple[np.ndarray, int]:
    t = np.arange(int(seconds * sr), dtype=np.float32) / float(sr)
    freq = 220.0 + (index % 7) * 35.0
    audio = 0.12 * np.sin(2.0 * math.pi * freq * t)
    audio += 0.02 * np.sin(2.0 * math.pi * (freq * 1.5) * t)
    return audio.astype(np.float32), sr


def gpu_snapshot() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        }
    except Exception as exc:
        return {"error": str(exc)}


def load_model(args: argparse.Namespace):
    from qwen_asr import Qwen3ASRModel

    if args.backend == "vllm":
        kwargs: dict[str, Any] = {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_inference_batch_size": max(args.batches),
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "mm_encoder_attn_backend": args.mm_encoder_attn_backend,
            "limit_mm_per_prompt": {"audio": 1},
            "dtype": args.dtype,
            "enforce_eager": args.enforce_eager,
        }
        if args.attention_backend:
            kwargs["attention_config"] = {"backend": args.attention_backend}
        return Qwen3ASRModel.LLM(model=args.model_id, **kwargs)

    import torch

    dtype = torch.bfloat16 if args.dtype in {"bf16", "bfloat16"} else torch.float16
    return Qwen3ASRModel.from_pretrained(
        args.model_id,
        device_map="cuda:0",
        torch_dtype=dtype,
        attn_implementation="sdpa",
        max_inference_batch_size=max(args.batches),
        max_new_tokens=args.max_new_tokens,
    )


def run_case(model: Any, batch: int, seconds: float, language: str, repeats: int, sr: int) -> dict[str, Any]:
    samples = [make_audio(seconds, sr, i) for i in range(batch)]
    latencies: list[float] = []
    errors: list[str] = []
    for _ in range(repeats):
        started = time.perf_counter()
        try:
            outputs = model.transcribe(
                audio=samples,
                language=None if language == "auto" else [language] * batch,
                return_time_stamps=False,
            )
            if len(outputs) != batch:
                errors.append(f"expected {batch} outputs, got {len(outputs)}")
        except Exception as exc:
            errors.append(str(exc))
            break
        latencies.append(time.perf_counter() - started)
    ok = len(latencies)
    total_audio_s = batch * seconds
    return {
        "batch": batch,
        "audio_seconds": seconds,
        "language": language,
        "repeats": repeats,
        "ok": ok,
        "failed": repeats - ok,
        "errors": errors[:3],
        "avg_s": statistics.mean(latencies) if latencies else 0.0,
        "p95_s": percentile(latencies, 95),
        "audio_s_per_s": (total_audio_s / statistics.mean(latencies)) if latencies else 0.0,
        "gpu": gpu_snapshot(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--mm-encoder-attn-backend", default="TORCH_SDPA")
    parser.add_argument("--attention-backend", default="TRITON_ATTN")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 96, 128])
    parser.add_argument("--lengths", type=float, nargs="+", default=[10.0, 30.0])
    parser.add_argument("--languages", nargs="+", default=["Vietnamese", "English", "auto"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = parse_args()
    started = time.perf_counter()
    model = load_model(args)
    load_s = time.perf_counter() - started
    cases = []
    for language in args.languages:
        for seconds in args.lengths:
            for batch in args.batches:
                case = run_case(model, batch, seconds, language, args.repeats, args.sample_rate)
                cases.append(case)
                print(
                    f"{args.model_id} {language} {seconds:g}s batch={batch} "
                    f"ok={case['ok']} failed={case['failed']} avg={case['avg_s']:.3f}s "
                    f"throughput={case['audio_s_per_s']:.1f}x gpu={case['gpu']}",
                    flush=True,
                )
                if case["failed"]:
                    break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "backend": args.backend,
                "model_id": args.model_id,
                "load_s": load_s,
                "args": vars(args) | {"output": str(args.output)},
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
