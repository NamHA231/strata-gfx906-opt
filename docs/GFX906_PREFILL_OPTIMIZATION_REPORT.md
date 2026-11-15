# BÁO CÁO TỐI ƯU HÓA PREFILL STRATA ENGINE TRÊN AMD INSTINCT MI50 (GFX906)

- **Nhánh:** `opt/gfx906-prefill-breakthrough` (base `c1610b09b`)
- **Phần cứng:** AMD Radeon Instinct MI50 32GB HBM2 (`gfx906:sramecc+:xnack-`, 60 CUs, không có MFMA/WMMA tensor core), AMD Ryzen 5 5600G, 64GB DDR4
- **Phần mềm:** ROCm 6.3.3 (`hipcc`, `rocblas`), Strata Engine `build-906-pr/strata`
- **Mô hình đo kiểm:** `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS` (`sc117-iq3xxs`, `kv int8`, `conversation-cache-mib 4096`, `vram-reserve-mib 1500`)

---

## 1. Bối cảnh & 4 điểm nghẽn chính của Prefill trên `gfx906`

Trước đợt tối ưu này, luồng prefill của `strata-coder.service` trên MI50 đạt:
- `~106` tokens: **`67.2 tok/s`**
- `~291` tokens: **`133.2 tok/s`**
- `~932` tokens: **`246.0 tok/s`**
- `~3928` tokens: **`371.0 tok/s`**

Phân tích cấu trúc kernel và runtime trên `gfx906` cho thấy 4 điểm nghẽn độc lập:
1. **Dense GEMM (`FP16` / `rocBLAS`):** bản build `STRATA_HIP_GFX906` chưa liên kết trực tiếp `rocblas` (`STRATA_ROCBLAS_AVAILABLE=1`) và thiếu bảng tuning `f16o_c32` cho hình học của `Qwen3.8-Flash` trên MI50.
2. **QSA Sparse Attention (`qsa_prompt_attn.cu`):** đường fallback `qsa_decode_attn_batch` trên `gfx906` chưa tận dụng tái sử dụng `Q` trong LDS, giảm cross-lane bằng warp-DPP và chưa gộp `gate_attn` vào epilogue của kernel attention (`qsa_prompt_attn_batch_gated`).
3. **MoE Fused `native_w11_kernel` (`IQ3_XXS`, `moe_fused_iq.cu`):** biến thể `ILi17` (`IQ3_XXS`) giữ quá nhiều thanh ghi tích lũy `nw_ab` và `float` cùng lúc trên `nt=0..1`, đẩy mức sử dụng lên `256 VGPRs` và gây tràn **`108 B/lane` scratch spill**.
4. **GDN Recurrence & Conv (`kernels.cu` / `prefill.cpp`):** chưa bật bộ cờ tối ưu `STRATA_GDN_HEAD=1`, `STRATA_GDN_PP=2`, `STRATA_GDN_NOY=1`, `STRATA_GDN_CONVL2=2` đã kiểm chứng parity trên `gfx906`.

---

## 2. Chi tiết kỹ thuật 4 đột phá + Tối ưu hạ tầng đi kèm

### Đột phá #1 — Dense GEMM `FP16` với `rocBLAS` MI50 Tuning Table
- **Thay đổi trong [CMakeLists.txt](../CMakeLists.txt):**
  - Thêm `find_package(rocblas CONFIG QUIET)`, định nghĩa `STRATA_ROCBLAS_AVAILABLE=1`, liên kết `roc::rocblas` vào `strata` khi `STRATA_HIP_GFX906` được bật và đồng bộ file tuning vào thư mục `tuning/`.
- **Bảng tuning:** [tools/hip/gfx906-rocblas-4.3.0.8ebd6c11.txt](../tools/hip/gfx906-rocblas-4.3.0.8ebd6c11.txt) gồm `154` dòng (`125` dòng có solution tối ưu cho MI50 `f16o_c32`).
- **Cấu hình runtime:** bật `STRATA_HIP_PROMPT_F16=1` và `STRATA_ROCBLAS_TUNING=.../gfx906-rocblas-4.3.0.8ebd6c11.txt`.
- **An toàn FP16:** kiểm tra live với `STRATA_F16_RANGE=1` trên `415`, `1399`, `6026`, `113` tokens cho `max |x| <= 103.9`, `max |y| <= 312.8`, **`0` giá trị vượt ngưỡng FP16 (`65504`) hay `NaN`**.

### Đột phá #2 — Flash-QSA Zero-Spill & Gated Epilogue cho `gfx906`
- **Thay đổi trong [src/kernels/cuda/qsa_prompt_attn.cu](../src/kernels/cuda/qsa_prompt_attn.cu):**
  - Cài đặt `gfx906_dpp_xor`, `qsa_flash_gfx906_kernel<bool GATE>` và `launch_flash_gfx906` (mặc định bật khi `n_q >= 32`, có thể tắt bằng `STRATA_QSA_FLASH_906=0`).
  - Nạp tile `Q` (`sq`) vào LDS (`12 KB`), đọc coalesced `K/V`, giảm bằng `__dpp` / `quad_perm` (`dpp_xor<1,2>`), đạt **`0` register spill** và **`0` scratch allocation**.
  - Đấu nối cả `qsa_prompt_attn_batch_gated` (gộp sigmoid `gate_attn` trực tiếp vào epilogue, bỏ kernel launch và đọc/ghi FP32 trung gian) lẫn `qsa_prompt_attn_batch`.
- **Hiệu năng & Parity:**
  - Microbench standalone trên MI50: **`1.27x`** tại `nq=928, ctx=2048` (`21.594 -> 16.976 ms`) và **`1.31x`** tại `nq=1960, ctx=3920` (`59.944 -> 45.821 ms`, `rel_scale=7.054e-07`).
  - Parity API đầy đủ độ rộng edge (`0..2048`, non-zero `ld16 = NH*HD + 64`): `max_diff=2.384e-06`, `rel_scale=6.734e-07` (non-gated) và `max_diff=9.737e-04`, `rel_scale=3.079e-04` (gated).

### Đột phá #3 — Xóa 100% Scratch Spill trong MoE `native_w11_kernel` (`IQ3_XXS`)
- **Thay đổi trong [src/prefill/moe_fused_iq.cu](../src/prefill/moe_fused_iq.cu):**
  - Trong nhánh `#if STRATA_NAT_W906` của `native_w11_kernel`, chia vòng lặp hàng kích hoạt `i0 = 0, 4` (4-row A blocking xuyên qua `nt=0..1`) thay vì giữ toàn bộ thanh ghi cùng lúc.
- **Kết quả ISA (`llvm-readelf --notes`):**
  - Biến thể `IQ3_XXS` (`ILi17`) giảm từ `256 VGPRs, 108 B/lane` spill xuống **`245..246 VGPRs, 0 B` spill**.
- **Hiệu năng & Parity (`prefill_fused_iq_test --only="IQ3_XXS / IQ4_NL" --chunks=928`):**
  - Bit-exact (`fused output bits hash 9bddea246f792077`), `MMQ path 29.588 ms` vs `fused 22.145 ms` (**`1.34x`**, min `21.995 ms`), pass parity.

### Đột phá #4 — Kích hoạt GDN Recurrence & Conv Fast Path
- **Cờ bật trong production config:** `STRATA_GDN_HEAD=1`, `STRATA_GDN_PP=2`, `STRATA_GDN_NOY=1`, `STRATA_GDN_CONVL2=2`.
- **Kiểm tra parity & microbench:**
  - `gdn_parity`: `gdn: 0 failures` (`o vs reference rel 2.001e-07`, `conv out rel 2.477e-08`, `state rel 3.736e-08`).
  - `gfx906_gdn_rec_head`: `FAILURES: 0`, `T=2048 variant 1 vs 0: 7.581 -> 3.718 ms`.

### Tối ưu hạ tầng & thử nghiệm đi kèm (12/11/2026)
- `src/prefill/moe_fused.cu`: bật đường `STRATA_FUSED_W11` trên `__gfx906__` dùng `__builtin_amdgcn_sdot4`.
- `src/prefill/moe_mmq.cu`: mở `STRATA_MMQ_TAB` & `STRATA_MMQ_DIRECT` cho `STRATA_HIP_GFX906`.
- `src/prefill/prefill.cpp` & `src/prefill/kernels.cu`: mặc định `STRATA_PF_TAIL_SKIP=7` cho non-resident tail stragglers trên `gfx906` kèm guard slot âm trong `moe_combine_kernel` / `moe_combine4_kernel`.
- `include/strata/platform/hip_host_lock.hpp` & `src/core/expert_source.cpp`: bổ sung `MADV_HUGEPAGE` + fallback mlock từng phần khi chạm `RLIMIT_MEMLOCK`.
- `src/kernels/cuda/native_moe.cu` & `src/core/verify.cpp`: bổ sung `STRATA_FUSE_HITS` và shared gate cho `gfx906` (tắt mặc định, xem `bench/results/2026-11-12-gfx906-fuse-hits/README.md`).
- `src/kernels/cuda/native_mmvq.cu`: thử nghiệm opt-in `STRATA_GFX906_MMVQ_ROWS=2` (mặc định giữ 1 row, xem `bench/results/2026-11-12-gfx906-mmvq-r2/README.md`).

---

## 3. Kết quả đo End-to-End Live (`strata-coder.service`)

Tất cả các phép đo dưới đây thực hiện với `cache_n = 0` (prefix duy nhất ở mỗi request, tắt toàn bộ cờ debug `STRATA_DBG_NAN` / `STRATA_F16_RANGE`):

| Nhóm độ dài prompt (Trước -> Sau) | Trước tối ưu (`tok/s`) | Sau tối ưu (`tok/s`) | Mức tăng | Thời gian prompt |
|---|---:|---:|---:|---:|
| Ngắn (`106 -> 118` tok) | `67.2` | **`86.5`** | **+28.7%** | `1364 ms` (`GPU 1024 ms`) |
| Nhỏ (`291 -> 272` tok) | `133.2` | **`148.5`** | **+11.5%** | `1832 ms` (`GPU 1531 ms`) |
| Vừa (`932 -> 895` tok) | `246.0` | **`296.4`** | **+20.5%** | `3020 ms` (`GPU 2652 ms`) |
| Dài (`3928 -> 3997` tok) | `371.0` | **`565.4`** | **+52.4% (`1.52x`)** | `7069 ms` (`GPU 6609 ms`) |

### Phân rã GPU Timeline (`STRATA_PREFILL_TIMING=1`, `3992` tokens, tổng GPU `6609 ms`)
- `gemm gate/up` (MoE fused): `2399 ms` (`36.3%`)
- `gdn` (Dense GEMM QKV/gate) + `gdn out proj` + `gdn recurrence` + `conv+gates`: `1101 + 377 + 264 + 22 = 1764 ms` (`26.7%`)
- `qsa attn` + `qsa proj` + `select/idx/prep`: `838 + 424 + 71 = 1333 ms` (`20.2%`)
- `hc read` (`down+silu 307`, `up 236`, `inject 74`, `mix 54`) + `hc write`: `671 + 76 = 747 ms` (`11.3%`)
- `router+shared`: `272 ms` (`4.1%`)

### Smoke Test chất lượng & dịch vụ sau restart sạch
- `systemctl --user is-active strata-coder.service` -> `active`, `GET /health` -> `200`.
- Request C++17 (`cache_n=0`, `reasoning_budget_tokens=96`, `temperature=0`): `finish_reason: stop`, decode **`45.1 tok/s`** (`148/162` draft accepted), đầu ra hàm `binary_search_lower` chính xác và mạch lạc.

---

## 4. Biến môi trường khuyến nghị cho Prefill + Decode MI50

```json
"env": {
  "HSA_OVERRIDE_GFX_VERSION": "9.0.6",
  "HIP_VISIBLE_DEVICES": "0",
  "STRATA_EXP_MODE": "8",
  "STRATA_PF_FUSED": "1",
  "STRATA_MMQ_DIRECT": "1",
  "STRATA_MMQ_GROUP_SORT": "1",
  "STRATA_PF_TAIL_SKIP": "7",
  "STRATA_HIP_PROMPT_F16": "1",
  "STRATA_ROCBLAS_TUNING": "/path/to/tools/hip/gfx906-rocblas-4.3.0.8ebd6c11.txt",
  "STRATA_HIP_SWIGLU_FUSED": "1",
  "STRATA_HIP_ROUTER_FAST": "1",
  "STRATA_GFX906_ATTN_QUERY_SWIZZLE": "0",
  "STRATA_GFX906_ATTN_REDUCE12": "1",
  "STRATA_GFX906_MMQ_MAX_T": "64",
  "STRATA_GFX906_TOPK_SPLIT": "1",
  "STRATA_LOOKUP_MAX_T": "0",
  "STRATA_GDN_TC": "0",
  "STRATA_GDN_HEAD": "1",
  "STRATA_GDN_PP": "2",
  "STRATA_GDN_NOY": "1",
  "STRATA_GDN_CONVL2": "2"
}
```
