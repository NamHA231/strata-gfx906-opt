# MI50 expert tile experiment, 12 November 2026

The verifier's native IQ3_S/Q2_0 expert kernel stages up to four entries of a routed expert in LDS (mode 8). Two gfx906-only prototypes staged one entry (mode 9) or two entries (mode 10). Both kept each entry's arithmetic and passed the benchmark's byte-for-byte output comparison against mode 8 on real GGUF experts. Both were slower and were removed from the source after measurement.

Hardware: AMD Instinct MI50 32 GB (gfx906), ROCm 6.3.3. Model: Qwen3.8-Flash-Next Coder IQ3_XXS, shard 1. Benchmark: `native_expert_bench`, layers 8 and 24, `BENCH_CAP=30`, 40 iterations, mode 8 as reference. The model service was left running, so other GPU work may affect small timing differences.

| Groups × entries | Layer | Mode 8 total | Candidate total | Output |
| --- | ---: | ---: | ---: | --- |
| 24 × 1 | 8 | 192.3 µs | mode 9: 222.8 µs | bitwise equal |
| 24 × 1 | 24 | 192.3 µs | mode 9: 222.4 µs | bitwise equal |
| 20 × 2 | 8 | 206.0 µs | mode 10: 209.9 µs | bitwise equal |
| 20 × 2 | 24 | 206.1 µs | mode 10: 231.5 µs | bitwise equal |
| 20 × 2 | 8 | 202.1 µs | mode 7: 234.1 µs | bitwise equal |
| 20 × 2 | 24 | 203.8 µs | mode 7: 238.5 µs | bitwise equal |
| 20 × 2 | 8 | 201.4 µs | mode 4: 529.3 µs | bitwise equal |
| 20 × 2 | 24 | 203.6 µs | mode 4: 546.7 µs | bitwise equal |

These microbenchmarks rule out smaller LDS entry tiles and modes 7/4 for this measured shape. They do not establish a hardware speed limit or predict every decode workload.
