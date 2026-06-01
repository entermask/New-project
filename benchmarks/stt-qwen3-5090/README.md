# Qwen3-ASR RTX 5090 Benchmark Notes

Target host used during implementation:

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB
- Driver: 570.169
- Python: 3.12.3

Observed setup results:

- `qwen-asr[vllm]` stable stack installed `torch==2.9.1+cu128`, `vllm==0.14.0`, `transformers==4.57.6`.
- vLLM stable fails with default FlashAttention on driver 570.169, but runs when forcing:
  - `attention_config.backend=TRITON_ATTN`
  - `mm_encoder_attn_backend=TORCH_SDPA`
  - `max_model_len=8192`
- vLLM nightly installed `torch==2.11.0+cu129`, `vllm==0.21.1rc1`, and CUDA 13 runtime libraries. It required `LD_LIBRARY_PATH` to include `site-packages/nvidia/cu13/lib`.
- vLLM nightly then failed on driver 570.169 with `CUDA driver version is insufficient for CUDA runtime version`.

Current recommendation for this exact host:

- Use `QWEN_ASR_BACKEND=vllm` with stable `qwen-asr[vllm]`.
- Set `QWEN_ASR_VLLM_ATTENTION_BACKEND=TRITON_ATTN`.
- Set `QWEN_ASR_VLLM_MM_ENCODER_ATTN_BACKEND=TORCH_SDPA`.
- Set `QWEN_ASR_VLLM_MAX_MODEL_LEN=8192` for short/medium high-throughput service.
- Set `QWEN_ASR_GPU_MEMORY_UTILIZATION=0.35` when loading both 0.6B and 1.7B in the same STT server.
- Set `QWEN_ASR_MAX_INFERENCE_BATCH_SIZE=32` as the latency/throughput sweet spot for 30s audio. Batch 64/96/128 works on 0.6B without OOM, but p95 latency jumps sharply and throughput is less stable.

0.6B vLLM direct benchmark artifact:

- JSON: `vllm-triton-sdpa-06b-5090-driver570.json`
- For 30s audio, batch 32 delivered ~2388 audio-s/s Vietnamese, ~2409 audio-s/s English, and ~2552 audio-s/s auto with ~0.42s p95 wall time.
- Batch 64+ did not OOM, but 30s p95 moved to ~0.94s-1.84s depending language and batch.

1.7B vLLM direct benchmark artifact:

- JSON: `vllm-triton-sdpa-17b-5090-driver570.json`
- For 30s audio, batch 32 delivered ~1460 audio-s/s Vietnamese, ~1903 audio-s/s English, and ~2319 audio-s/s auto.
- Batch 64+ did not OOM, but forced-language p95 moved to ~1.0s-2.4s.

Compact result table: `vllm-triton-sdpa-summary.md`.

Dual-model startup check:

- `QWEN_ASR_GPU_MEMORY_UTILIZATION=0.35`
- Both `Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B` loaded in one process on the RTX 5090.
- vLLM reported max concurrency for 8192-token requests of ~6.57x for 0.6B and ~3.89x for 1.7B after both engines reserved KV/cache.

Run full API benchmark after the STT server is up:

```bash
python scripts/benchmark_stt_qwen.py \
  --base-url http://127.0.0.1:6006 \
  --token "$API_TOKEN" \
  --audio short.wav medium.wav long.wav \
  --models 0.6b 1.7b \
  --languages vi en auto \
  --concurrency 1 2 4 8 16 32 64 96 128
```
