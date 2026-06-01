# Qwen3-ASR Direct Batch Benchmark

Host: RTX 5090 32GB, driver 570.169, Python 3.12.3.
Backend: `Qwen3ASRModel.from_pretrained`, `torch==2.9.1+cu128`, `dtype=bfloat16`, `attn_implementation=sdpa`, `max_new_tokens=256`.

This is the earlier transformers fallback benchmark. vLLM was later made usable on this driver by forcing language attention to `TRITON_ATTN` and MM encoder attention to `TORCH_SDPA`; see `vllm-triton-sdpa-06b-5090-driver570.json`.

Initial vLLM attempts failed on this driver:

- Stable vLLM failed during audio encoder FlashAttention/PTX profiling.
- Nightly vLLM loaded farther but failed with `CUDA driver version is insufficient for CUDA runtime version`.

Sweet spot: `QWEN_ASR_MAX_INFERENCE_BATCH_SIZE=32`.

Reason: 0.6B peaks at batch 32 on 30s audio and batch 64 only helps short 10s audio. 1.7B batch 64 is only ~2% faster on 30s audio but worse on 10s audio and uses ~7.4GB more VRAM than batch 32.

| model | audio each | batch | throughput audio-s/s | avg wall s | p95 wall s | used VRAM GB |
|---|---:|---:|---:|---:|---:|---:|
| 0.6b | 10s | 1 | 20.84 | 0.480 | 0.767 | 2.39 |
| 0.6b | 10s | 2 | 63.48 | 0.315 | 0.338 | 2.41 |
| 0.6b | 10s | 4 | 106.73 | 0.375 | 0.388 | 2.41 |
| 0.6b | 10s | 8 | 143.36 | 0.558 | 0.572 | 2.74 |
| 0.6b | 10s | 16 | 219.60 | 0.729 | 0.859 | 3.42 |
| 0.6b | 10s | 32 | 187.91 | 1.703 | 1.800 | 4.76 |
| 0.6b | 10s | 64 | 231.41 | 2.766 | 3.210 | 7.44 |
| 0.6b | 30s | 1 | 110.70 | 0.271 | 0.299 | 7.47 |
| 0.6b | 30s | 2 | 189.64 | 0.316 | 0.340 | 7.47 |
| 0.6b | 30s | 4 | 184.64 | 0.650 | 0.696 | 7.47 |
| 0.6b | 30s | 8 | 230.48 | 1.041 | 1.076 | 7.47 |
| 0.6b | 30s | 16 | 261.63 | 1.835 | 1.971 | 7.47 |
| 0.6b | 30s | 32 | 324.67 | 2.957 | 3.031 | 11.16 |
| 0.6b | 30s | 64 | 304.81 | 6.299 | 6.759 | 18.55 |
| 1.7b | 10s | 1 | 19.73 | 0.507 | 0.540 | 5.04 |
| 1.7b | 10s | 2 | 50.39 | 0.397 | 0.412 | 5.06 |
| 1.7b | 10s | 4 | 83.93 | 0.477 | 0.494 | 5.06 |
| 1.7b | 10s | 8 | 122.41 | 0.654 | 0.684 | 5.06 |
| 1.7b | 10s | 16 | 142.07 | 1.126 | 1.186 | 5.73 |
| 1.7b | 10s | 32 | 196.96 | 1.625 | 1.962 | 7.07 |
| 1.7b | 10s | 64 | 162.32 | 3.943 | 4.232 | 9.75 |
| 1.7b | 30s | 1 | 74.28 | 0.404 | 0.408 | 9.77 |
| 1.7b | 30s | 2 | 103.40 | 0.580 | 0.629 | 9.77 |
| 1.7b | 30s | 4 | 160.35 | 0.748 | 0.766 | 9.77 |
| 1.7b | 30s | 8 | 215.19 | 1.115 | 1.361 | 9.77 |
| 1.7b | 30s | 16 | 307.41 | 1.561 | 1.768 | 9.77 |
| 1.7b | 30s | 32 | 304.28 | 3.155 | 3.349 | 13.47 |
| 1.7b | 30s | 64 | 310.97 | 6.174 | 6.582 | 20.86 |
