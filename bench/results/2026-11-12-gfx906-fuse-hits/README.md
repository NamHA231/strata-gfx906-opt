# gfx906 MoE hit fusion, 12 November 2026

The change makes `STRATA_FUSE_HITS=1` available on gfx906. It folds the GPU hit-row add into the MoE combine and also handles the shared-expert gate. The flag remains off by default on gfx906.

## Hardware and setup

- AMD Instinct MI50 32 GB, gfx906, ROCm 6.3.3.
- Qwen3.8-Flash-Next Coder IQ3_XXS, Strata engine 0.1.57, `--spec 3 --spec-min-p 0.5`, int8 KV, resident-expert cache.
- Same candidate executable for both arms (`8beae8f1e4e442fb6d4b64dbfd8b6885e16c43ef8a6b082dabd3431c8b93e3ef` SHA-256). Arm 0 sets `STRATA_FUSE_HITS=0`; arm 1 sets it to `1`. Both set `STRATA_FUSE_WAIT=0` and `STRATA_DECODE_TIMING=1`.
- Greedy sampling, seed 906, 384 generated tokens per measured request, two requests per prompt length. Each arm starts a fresh engine and runs one 32-token warm-up first. Prompt IDs are stored here. `run_arm.py` records the native protocol, engine log, command, environment, and per-request results.

## Results

| Prompt | Fusion off, tokens/s | Fusion on, tokens/s | Mean difference |
| --- | --- | --- | --- |
| 256 tokens | 46.216, 46.512 | 47.080, 45.973 | +0.35% |
| 32,768 tokens | 42.424, 41.366 | 42.445, 39.970 | -1.64% |

The fused GPU kernel matched the actual unfused add-plus-combine pipeline byte-for-byte in 160 MI50 test cases across 1, 2, 3, 4, and 8 tokens, four widths, and gated and ungated modes (`parity.txt`). End-to-end output was **not** identical: the two arms first differed after 126 and 95 generated tokens at the short prompt, and after 32 and 16 at the long prompt. Repeated requests within the same arm also differed. Draft acceptance and cached expert counts varied. This A/B run does not establish an end-to-end speed improvement or output parity.

The engine log reported roughly 34-39 ms of GPU-reach wait in a 41-46 ms verifier window, while the removed kernel is one launch per MoE layer. The measured differences are small beside run-to-run variation. The kernel fusion should remain opt-in pending a controlled, fixed-continuation benchmark. The service's original Strata executable was restored after the run.
