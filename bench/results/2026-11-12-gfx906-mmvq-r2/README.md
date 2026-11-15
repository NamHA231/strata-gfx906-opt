# MI50 wave64 MMVQ rows, 12 November 2026

An opt-in gfx906 kernel variant makes one wave compute two adjacent Q8_0 output rows (`STRATA_GFX906_MMVQ_ROWS=2`). The default remains one row. Other quantization formats always keep the original path: the full MMVQ parity test found bit differences at wide T when the two-row variant was applied to Q4_0, Q6_K, or IQ4_XS. The Q8_0-only variant passed `mmvq_multi_parity` and the direct R1/R2 byte comparison in `mmvq_wave_rows` on MI50.

Hardware: AMD Instinct MI50 32 GB, gfx906, ROCm 6.3.3. Model: Qwen3.8-Flash-Next Coder IQ3_XXS, Strata engine 0.1.57, speculative T≤3, int8 KV. Candidate executable SHA-256: `b30af8d1e7680b566f2feec02372ef1c71dfd439d4eb7060fd5908e497bf3673`. All runs kept `STRATA_FUSE_HITS=0` and `STRATA_FUSE_WAIT=0`.

## Kernel test

The Q8_0 test used identical random weight and Q8_1 activation bytes for each pair. R1 and R2 produced bitwise identical outputs.

| Shape | T | R1 | R2 |
| --- | ---: | ---: | ---: |
| 2560 × 2560 | 1 | 16.57 µs | 15.59 µs |
| 2560 × 2560 | 3 | 31.86 µs | 18.70 µs |
| 2560 × 640 | 3 | 10.89 µs | 7.96 µs |
| 640 × 2560 | 3 | 12.85 µs | 8.82 µs |

These are isolated GPU event timings while the model service was running. The full model does not reproduce this gain reliably.

## Full model

Native server protocol, same candidate binary and public prompts, two requests of 384 generated tokens per arm after warm-up:

| Prompt | R1 tokens/s | R2 tokens/s | Mean R2 difference |
| --- | --- | --- | ---: |
| 256 tokens | 47.417, 46.056 | 47.475, 47.840 | +1.97% |
| 32,768 tokens | 42.426, 39.943 | 43.697, 43.026 | +5.29% |

The freely generated token sequences differed after 126/95 tokens on the short prompt and 32/16 on the long prompt. Draft acceptance also varied. These numbers are not a controlled speed comparison.

CLI `--spec-follow` then forced the same 384-token continuation on the 32,768-token prompt, with adaptation off. The runs were ordered R1→R2 and then R2→R1. All four wrote the same 183 verify-window hashes byte-for-byte.

| Order | R1 | R2 |
| --- | ---: | ---: |
| R1→R2 | 42.26 tokens/s | 40.68 tokens/s |
| R2→R1 | 42.21 tokens/s | 43.16 tokens/s |

R1 averaged 42.24 tokens/s; R2 averaged 41.92 tokens/s (−0.76%). R2 varied much more between runs. This experiment does not establish an end-to-end speedup, so the default stays R1 and the running service uses its original executable.

The shard's GGUF tensor header contains 36 `ssm_out.weight` and 12 `attn_output.weight` matrices of type 8 (Q8_0); many other dense projections are Q4_K, Q6_K, IQ4_XS, or custom type 30. The R2 variant therefore accelerates only a subset of the 300 native projection matrices reported at model load. This is one reason an isolated Q8_0 kernel gain cannot be extrapolated to the full decode path.

Raw protocol, logs, manifests, prompts, continuations, and window hashes are in [`../2026-11-12-gfx906-fuse-hits`](../2026-11-12-gfx906-fuse-hits/). The separate kernel trace in [`../2026-11-12-gfx906-expert-tiling`](../2026-11-12-gfx906-expert-tiling/) ranked `native_mmvq_wave_kernel` among the expensive decode kernels, but tracing altered wall time; it was used to choose this experiment, not to claim a production throughput.
