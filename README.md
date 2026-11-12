# Strata GFX906 Opt (AMD Instinct MI50 / gfx906)

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-AMD_GCN_5.1_(gfx906)-red?style=for-the-badge&logo=amd" alt="AMD gfx906">
  <img src="https://img.shields.io/badge/Target_GPU-Instinct_MI50_32GB_HBM2-blue?style=for-the-badge&logo=amd" alt="AMD MI50">
  <img src="https://img.shields.io/badge/ROCm-6.3.3_(HIP)-purple?style=for-the-badge" alt="ROCm 6.3.3">
  <img src="https://img.shields.io/badge/Decode_Speed-39_--_45+_tok/s-success?style=for-the-badge" alt="Decode Speed">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
</p>

An ultra-optimized fork of the **Strata Engine** engineered specifically for the **AMD Radeon Instinct MI50 (32GB HBM2, `gfx906` / Vega 20)** running **ROCm 6.3.3 / HIP**. 

This fork resolves the severe decode bottlenecks and speculative MoE thrashing in standard Strata, accelerating large MoE models (such as `Qwen3.8-Flash-Next` 27B / 3.8B active) from **26–32 tok/s** to **39–45+ tok/s** across short, medium, and extreme long-context (40K+ tokens) workloads.

---

## 🚀 Key Highlights & Breakthroughs

1. **Bit-Exact Carry-Free `nib_neg4` ISA Kernel:**
   - Implements carry-free two's-complement negation for positive codebook bytes on AMD GCN 5.1 (`gfx906`), replacing ~24 emulated `__vcmpne4` + `__vsub4` VALU instructions with a single 24-bit multiply `__umul24(n, 0x204081u)` + bit logic (only 5 VALU ops + `v_dot4_i32_i8`).
   - Achieves 100% bit-exact parity across `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, and `IQ3_S`.
2. **Wave64 Paired Native MMVQ:**
   - Optimized for 64-wide wavefronts with a full 64-lane cross-wave reduction (`__shfl_xor(v, off, 64)`).
   - Computes dual matrix weights in a single pass over activation blocks (`PAIR=true`), cutting activation fetch pressure on HBM2 in half.
3. **LDS Grid Staging MMVQ:**
   - Stages `iq3s_grid` codebooks and bit masks into Local Data Share (LDS), eliminating cache bank conflicts during expert MMVQ decode.
4. **Sliding-Window MTP Drafter (`--mtp-window 4096`):**
   - Constrains dense attention KV scanning in the speculative MTP Drafter to a sliding 4,096-token window, slashing draft time at long context with zero degradation in draft accuracy.
5. **Full Wave64 Alignment & NaN Fixes:**
   - Rectifies CUDA 32-lane warp assumptions (`STRATA_GDN_CONVL2=0`, `STRATA_GDN_TC=0`), eliminating the catastrophic `non-finite logits (#879)` crash at long context lengths.
   - Tailored specifically for the 64-wide wavefronts of AMD Vega 20 (`gfx906`).

---

## 📊 Benchmark Results

Measured on **AMD Radeon Instinct MI50 32GB HBM2** + **AMD Ryzen 5 5600G** + **64GB DDR4-3200** via OpenAI-compatible API (`/v1/chat/completions`) using `Qwen3.8-Flash-Next-Coder-IQ3_XXS` (`max_tokens = 1200`):

| Scenario / Prompt Type | Context Length | Generation Phase | Decode Speed (`tok/s`) | Total Request Avg | Draft Acceptance | VRAM Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Upstream 0.1.57 (Unoptimized)** | Short (~100 tok) | Mixed | `26.6 – 28.5 tok/s` | `27.5 tok/s` | `54.4%` (`1.71` tok/w) | `96.2%` |
| **Upstream 0.1.57 (Hermes Real 41K)** | `41,115` tok | Mixed | `32.9 tok/s` | `32.9 tok/s` | `47.0%` (`2.06` tok/w) | `96.6%` (46 misses/w) |
| **`Strata-gfx906-opt` (Short Prompt)** | `92 → 798` tok | Thinking (`691` tok)<br>Writing (`494` tok) | **`41.9 tok/s`**<br>**`43.9 tok/s`** | **`42.7 tok/s`** | **`64.3%`** (`2.42` tok/w)<br>**`65.7%`** (`2.55` tok/w) | **`98.1%`**<br>**`99.1%`** |
| **`Strata-gfx906-opt` (Multi-Turn 11K)** | `10,511 → 11,217` tok | Thinking (`688` tok)<br>Writing (`494` tok) | **`39.3 tok/s`**<br>**`43.2 tok/s`** | **`40.9 tok/s`** | **`58.3%`** (`2.21` tok/w)<br>**`66.0%`** (`2.63` tok/w) | **`99.6%`**<br>**`99.1%`** |
| **`Strata-gfx906-opt` (Extreme Long 39.5K)** | `38,850 → 39,556` tok | Thinking (`689` tok)<br>Writing (`494` tok) | **`38.8 tok/s`**<br>**`45.6 tok/s`** | **`41.4 tok/s`** | **`59.6%`** (`2.14` tok/w)<br>**`73.0%`** (`2.84` tok/w) | **`99.8%`**<br>**`99.3%`** |

> Detailed technical logs, regression formulas, and step-by-step diagnosis are documented in [`docs/GFX906_COMPLETE_OPTIMIZATION_REPORT.md`](docs/GFX906_COMPLETE_OPTIMIZATION_REPORT.md).

---

## 🛠️ Build & Installation

### Requirements
- **OS:** Ubuntu 22.04 / 24.04 LTS (Linux x86_64)
- **Toolchain:** ROCm 6.3.x (ROCm 6.3.3 recommended) with HIP runtime
- **Compiler:** `hipcc` / `clang++` with OpenMP and C++20 support
- **Python:** Python 3.10+ with `fastapi`, `uvicorn`, `torch` (optional)

### Compilation
```bash
git clone https://github.com/NamHA231/strata-gfx906-opt.git
cd strata-gfx906-opt

# Configure build for gfx906 architecture
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_ARCHITECTURES="gfx906" \
  -DSTRATA_USE_HIP=ON \
  -DCMAKE_C_COMPILER=/opt/rocm-6.3.3/bin/hipcc \
  -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.3/bin/hipcc

# Build engine binaries
cmake --build build --target strata -j$(nproc)
```

---

## ⚙️ Serving Configuration

Here is the production configuration for running `Qwen3.8-Flash-Next-Coder-IQ3_XXS` on a single 32GB MI50 card:

### `config.json`
```json
{
  "exe": "/path/to/strata-gfx906-opt/build/strata",
  "port": 8080,
  "backend": "hip",
  "parallel": 1,
  "gpu": [0],
  "env": {
    "HSA_OVERRIDE_GFX_VERSION": "9.0.6",
    "HIP_VISIBLE_DEVICES": "0",
    "STRATA_WATCHDOG_S": "180",
    "STRATA_COMMIT_SYNC": "1",
    "STRATA_EXP_MODE": "8",
    "STRATA_PF_FUSED": "1",
    "STRATA_HIP_SWIGLU_FUSED": "1",
    "STRATA_HIP_ROUTER_FAST": "1",
    "STRATA_GFX906_ATTN_QUERY_SWIZZLE": "0",
    "STRATA_GFX906_ATTN_REDUCE12": "1",
    "STRATA_GFX906_MMQ_MAX_T": "64",
    "STRATA_GFX906_TOPK_SPLIT": "1",
    "STRATA_LOOKUP_MAX_T": "0",
    "STRATA_NV_CHUNK": "0",
    "STRATA_GDN_TC": "0",
    "STRATA_GDN_CONVL2": "0",
    "STRATA_ENGINE_NICE": "0",
    "STRATA_PCIE_FRAC_DEFAULT": "old",
    "STRATA_OLD_Q80_SCALED": "1",
    "STRATA_ROUTE_RESIDENT": "0.50",
    "STRATA_ROUTE_TAIL_SKIP": "4",
    "STRATA_DRAFT_TAIL_SKIP": "2",
    "STRATA_FUSE_WAIT": "1",
    "STRATA_FUSE_HITS": "1",
    "STRATA_DECODE_TIMING": "1"
  },
  "args": [
    "--pack", "/path/to/packs/sc117-iq3xxs",
    "--native", "/path/to/models/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf",
    "--ple-gguf", "/path/to/models/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "auto",
    "--spec", "4",
    "--spec-min-p", "0.10",
    "--mtp", "/path/to/Strata-data/mtp/rt",
    "--mtp-window", "4096",
    "--lookup-max-t", "0",
    "--max-context", "131072",
    "--kv", "int8",
    "--kv-resident", "0",
    "--vram-reserve-mib", "1500",
    "--conversation-cache-mib", "4096",
    "--conversation-cache-slots", "4",
    "--resident-experts",
    "--tail-role-token", "8678"
  ]
}
```

### Starting the Server
```bash
python3 -m serve.server --engine strata --config config.json --host 0.0.0.0 --port 8080
```

---

## 📖 In-Depth Documentation

- 📑 [**Complete Optimization Report (Vietnamese)**](docs/GFX906_COMPLETE_OPTIMIZATION_REPORT.md): Full narrative from initial bug diagnosis, mathematical modeling, kernel design, to verified benchmarks.
- 🔬 [**Decode Speed Diagnosis & Log Regression**](docs/GFX906_DECODE_SPEED_DIAGNOSIS_0211.md): Technical breakdown of the 398,999 verify rounds from real Hermes traffic.
- ⚡ [**Single MI50 Tuning Guide**](docs/GFX906_SINGLE_MI50_IQ3_S.md): VRAM cache sizing, PCIe bandwidth tuning, and pack preparation.

---

## 📜 License & Acknowledgments

This project is licensed under the [MIT License](LICENSE). Built upon the foundational architecture of [Strata](https://github.com/Niko1221/Strata). Optimized with dedication for AMD ROCm and open-source local AI infrastructure.
