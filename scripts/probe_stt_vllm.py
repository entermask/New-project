import argparse
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--mm-encoder-attn-backend", default="TORCH_SDPA")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--sleep", type=int, default=5)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    print("probe start", flush=True)
    from qwen_asr import Qwen3ASRModel

    print("qwen_asr import ok", flush=True)
    kwargs = {}
    if args.attention_backend:
        kwargs["attention_config"] = {"backend": args.attention_backend}

    model = Qwen3ASRModel.LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_inference_batch_size=args.batch,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        mm_encoder_attn_backend=args.mm_encoder_attn_backend,
        limit_mm_per_prompt={"audio": 1},
        enforce_eager=args.enforce_eager,
        **kwargs,
    )
    print(f"loaded backend={model.backend}", flush=True)
    for i in range(args.sleep):
        print(f"alive {i}", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
