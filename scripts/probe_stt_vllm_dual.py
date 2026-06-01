#!/usr/bin/env python3
import argparse
import os


def load(repo_id: str, args: argparse.Namespace):
    from qwen_asr import Qwen3ASRModel

    print(f"loading {repo_id}", flush=True)
    model = Qwen3ASRModel.LLM(
        model=repo_id,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_inference_batch_size=args.batch,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        mm_encoder_attn_backend=args.mm_encoder_attn_backend,
        attention_config={"backend": args.attention_backend},
        limit_mm_per_prompt={"audio": 1},
        dtype=args.dtype,
    )
    print(f"loaded {repo_id}", flush=True)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--attention-backend", default="TRITON_ATTN")
    parser.add_argument("--mm-encoder-attn-backend", default="TORCH_SDPA")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = parse_args()
    models = [
        load("Qwen/Qwen3-ASR-0.6B", args),
        load("Qwen/Qwen3-ASR-1.7B", args),
    ]
    print(f"loaded_count={len(models)}", flush=True)


if __name__ == "__main__":
    main()
