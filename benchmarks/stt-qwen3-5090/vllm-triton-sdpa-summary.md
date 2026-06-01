# Qwen3-ASR vLLM RTX 5090 Summary

Host:

- GPU: RTX 5090 32GB
- Driver: 570.169
- Stack: `qwen-asr[vllm]`, `torch==2.9.1+cu128`, `vllm==0.14.0`
- Required vLLM overrides: `attention_config.backend=TRITON_ATTN`, `mm_encoder_attn_backend=TORCH_SDPA`, `max_model_len=8192`
- Dual-model startup: use `gpu_memory_utilization=0.35` so 0.6B and 1.7B can both be resident.

Conclusion:

- Default production batch should stay `32`.
- Batch `64`, `96`, and `128` do not OOM on the direct benchmark, but p95 latency jumps sharply for normal forced-language workloads.
- Use batch `64+` only for offline bulk jobs where latency does not matter.
- The table below was measured one model at a time with `gpu_memory_utilization=0.7`; dual-model startup was separately verified at `0.35`.

30s audio results:

| model | language | batch | avg s | p95 s | audio-s/s | failed |
|---|---|---:|---:|---:|---:|---:|
| 0.6B | Vietnamese | 32 | 0.402 | 0.420 | 2388 | 0 |
| 0.6B | Vietnamese | 64 | 0.916 | 0.941 | 2095 | 0 |
| 0.6B | Vietnamese | 128 | 1.767 | 1.838 | 2174 | 0 |
| 0.6B | English | 32 | 0.399 | 0.421 | 2409 | 0 |
| 0.6B | English | 64 | 1.004 | 1.034 | 1913 | 0 |
| 0.6B | English | 128 | 1.928 | 1.941 | 1992 | 0 |
| 0.6B | auto | 32 | 0.376 | 0.414 | 2552 | 0 |
| 0.6B | auto | 64 | 0.964 | 1.014 | 1991 | 0 |
| 0.6B | auto | 128 | 1.387 | 1.474 | 2770 | 0 |
| 1.7B | Vietnamese | 32 | 0.658 | 0.752 | 1460 | 0 |
| 1.7B | Vietnamese | 64 | 1.438 | 1.628 | 1336 | 0 |
| 1.7B | Vietnamese | 128 | 2.379 | 2.387 | 1614 | 0 |
| 1.7B | English | 32 | 0.505 | 0.544 | 1903 | 0 |
| 1.7B | English | 64 | 1.005 | 1.035 | 1910 | 0 |
| 1.7B | English | 128 | 2.126 | 2.224 | 1806 | 0 |
| 1.7B | auto | 32 | 0.414 | 0.435 | 2319 | 0 |
| 1.7B | auto | 96 | 0.951 | 0.975 | 3030 | 0 |
| 1.7B | auto | 128 | 1.757 | 2.086 | 2186 | 0 |

Artifacts:

- `vllm-triton-sdpa-06b-5090-driver570.json`
- `vllm-triton-sdpa-17b-5090-driver570.json`
