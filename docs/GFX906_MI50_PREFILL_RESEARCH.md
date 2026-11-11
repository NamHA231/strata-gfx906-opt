# Prefill trên AMD MI50 (gfx906) — toàn cảnh tài liệu & thuật toán từ mạng, loại trừ CUDA

> Tổng hợp nghiên cứu: 21/10/2026. Nguồn: wiki-gfx906 (skyne98), LLVM/AMD docs, các fork llama.cpp-gfx906,
> fork vLLM gfx906, AMD-Ecosystem PR sets, papers (Sarathi-Serve, roofline), và đối chiếu với
> `Strata-gfx906-opt/docs/GFX906_PREFILL_METHODS.md` (số đo riêng của dự án).
> Mọi mục CUDA hoặc yêu cầu MFMA/WMMA được loại — và ghi rõ vì sao không tồn tại trên card này.

---

## 1. CỐT LỖI — suy từ cấu trúc MI50, không suy từ mô hình

MI50 = GCN5/Vega 7nm (`gfx906`), **không phải CDNA**. Tài liệu gốc (ROCm hw-specs, LLVM AMDGPUUsage,
Vega 7nm Shader ISA) xác nhận:

| Tài nguyên gfx906 | Giá trị | Hệ quả cho prefill |
|---|---|---|
| CU / wave | 60 CU (MI50), 64 (MI60); wavefront **64 lane** | mọi tile phải bội 64; kernel thiết kế warp32 phí 1 nửa lane |
| SIMD | 4×SIMD16/CU, tối đa 40 wave/CU | occupancy bị chốt bởi VGPR (256 KiB/CU) và LDS, không phải số wave slot |
| LDS | 64 KiB/CU, 32 bank ×4 B | bank conflict là limiter bậc nhất; tile FP16 32×32 = 2 KiB → chứa dư |
| L2 / L1 | 4 MiB L2; 16 KiB vector L1/CU | L2 dùng được làm "L2 tile reuse" nếu lịch wave đủ locality |
| HBM2 | ~1.0 TB/s danh định (ECC bật tụt nhẹ) | dense prefill không thiếu byte; **MoE streaming thì PCIe mới là nút byte** |
| **Ma trận** | **không MFMA, không WMMA, không tensor core** (`v_mfma*` chỉ có từ gfx908; WMMA từ RDNA1) | mọi matmul đi qua ống FMA thường — trần FLOP = trần FMA |
| **Dot nguyên** | LLVM gfx906 assembler **có** `v_dot4_i32_i8`, `v_dot8_i32_i4`, `v_dot2_f32_f16` | xem mục 5 — đây là cánh cửa bị bỏ sót |
| cp.async / s_delay_alu / s_clause | **không có** | mọi load vào LDS là VMEM→reg→LDS; latency chỉ giấu được bằng ILP + `s_waitcnt` đặt tay + occupancy |
| Đỉnh FP32 / FP16 | MI50: 13.4 / 26.8 TFLOPS danh định; nhưng `v_pk_fma_f16` packed **cùng nhịp** với FMA32 trên Vega | fp16 packed **không** mua thêm FLOP (trùng đo đạt trên máy này) |

### Công thức cốt lỗi

Prefill = **GEMM hàng loạt theo batch token** (`M = T` token, K = hidden, N = projection) + attention.
Với `T ≥ 128`, arithmetic intensity cao → **compute-bound**. Tốc độ tối đa suy từ card, không từ model:

```
tok/s ≤ min( FMA_đỉnh_card / MAC_per_token ,  PCIe_BW / bytes_cần_kéo_per_token )
```

Strata đã đo và khớp hai chiều độc lập (docs/GFX906_PREFILL_METHODS.md mục 5): GEMM FP16 cô lập đạt
**93–99% đỉnh card (~7.4 TFLOPS nội bộ của Strata)** tại T≥512 ⇒ toàn bộ room còn lại của prefill
nằm ở **thời gian ngoài GEMM** (dequant, gather, launch, PCIe wait) và ở **chính trần FMA nếu thay
bằng ống dot-int** (mục 5 dưới — chưa ai chứng minh là đóng).

Ba điều cấu trúc chi phối mọi thuật toán trên card này:
1. **Không có khối gia tốc ma trận** ⇒ tiết kiệm *số lệnh FMA* = tiết kiệm thời gian; lượng tử hoá
   GGUF tự nó **mua byte, không mua FLOP** (khi vẫn nhân bằng FMA32).
2. **Không cp.async** ⇒ mọi thiết kế LDS phải tự lo staging: register double-buffer + ILP + `s_waitcnt`.
3. **Wave64** ⇒ reduction phải là butterfly 64 lane (một `v_lshl`+`v_or` × 6 exchange), không phải
   cây 32 lane nhân bản từ CUDA; DPP/ds_bpermute nhanh hơn LDS exchange (đo: 1.78 T vs 0.91 T xchg/s).

---

## 2. DANH MỤC TOÀN BỘ CÁCH CÓ TRÊN MẠNG (không CUDA)

### A. llama.cpp/HIP backend — 4 đường mul-mat gốc
| Đường | Cơ chế | Tiền kiện | Trên MI50 |
|---|---|---|---|
| **MMVQ** (mul_mat_vec_q) | 1 block/wave, dot qua `vec_dotq` | batch=1 | decode, không phải prefill |
| **MMQ** (mul_mat_q) | weight nén × act **q8_1**, LDS tile 2D, bảng tile/arch | tile table đúng arch | có, nhưng bảng tile gốc tune theo warp32 → các fork sửa lại |
| **MMF** | dequant full rồi FMA | VRAM | chậm, fallback |
| **cuBLAS/hipBLASLt** | dequant→FP16 + HGEMM | lib có Tensile gfx906 | **đường mạnh nhất đã chứng minh** (93–99% đỉnh, số Strata) |

### B. Các fork llama.cpp gfx906 trên mạng (mỗi fork một nhóm thuật toán)
- **iacopPBK/llama.cpp-gfx906** (→ kế tục: `mxxm-t/mx-llama.cpp`), cấu trúc thư mục `ggml-cuda/gfx906/`:
  *DPP warp reductions*; **Q8_0 MMQ software-pipelining**; prefetch helpers; MMVQ warp-cooperative
  Q4_0/Q4_1/Q8_0; **half-warp dispatch fix cho wavefront64** (sửa `mmq.cu`/`mmid.cu` — MoE);
  MXFP4 load pipeline + e8m0 convert; `v_perm` latency-hiding cho MMVQ; **Q8 cross-op cache**
  (q8_1 activation cache dùng lại giữa các op = bỏ qua lượng tử hoá lại); epilogue DPP Q8_1.
  Đeffnote: hiệu năng **tăng theo power limit qua UPP** (`SCRIPT_overclock_upp_MI50.sh`) — nhưng
  máy này đã đo thật: 105W→178W chỉ +1.5% pp (compute-bound thật, không nghẽn điện).
- **eslowney/llama.cpp-gfx906**: head-dim 128 (Qwen3-30B), **FA kernel PP +5–11%** (lãi tăng theo
  ctx dài), 32K ctx trên 32 GB.
- **fuutott/llama.cpp-gfx906**: DPP reductions, vectorized quant kernels, Q8 flash attention
  **+18%** tổng inference.
- **bearqq/llama.cpp-gfx906** ("GLM and I created…", level1techs): **small-Q Flash Attention path**
  viết lại + **adaptive native/convert selection** → **+14.1% PP batch đầu, +9% fill ctx dài**;
  tác giả tự thừa nhận một số tối ưu (MTP) từng phản tác dụng khi mainline đổi — bài học:
  phải A/B theo build cùng ngày.
- **ML-gfx906 (mixa3607)** — nguyên stack gfx906 được bảo trì: apt repo + docker
  (`llama.cpp-gfx906`, vLLM, rocm-toolkit-gfx906 = ROCm 7.x cho gfx906, `amd-memory-tweak` chỉnh
  timing HBM2 qua debugfs, `amd-tuning` UPP). **Họ đang build chính Strata**: release
  20261020033835 có entry "Strata: `max-ilp` compile arg, +2–8%" — bằng chứng ILP compile flags
  là lever có thật trên stack gfx906 hiện trường.

### C. vLLM trên gfx906 (3 fork: kxb520/vllm-mi50, noshitcoding/vllm-mi50, btbtyler09 docker; kausik/vllm-gfx906)
Tất cả converge vào cùng một chẩn đoán — chính là cốt lỗi mục 1:
- `tl.dot` (Triton), Composable Kernel, AITer kernels đều sinh **MFMA → ILLEGAL_INSTRUCTION** trên
  gfx906 ⇒ tắt toàn bộ (`VLLM_ROCM_USE_AITER=0`, `AITER_MOE/GEMM/ROPE=0`), redirect
  **TRITON_ATTN → ROCM_ATTN**, rocBLAS phải có **Tensile `.hsaco` gfx906 thật** (lấy từ release
  rocBLAS Dez 2024, không phải wheel pip).
- **Thuật prefill đáng chú ý — "Direct K/V Prefill"**: khi cache còn trống (prompt đầu), **bỏ qua
  hoàn toàn PagedAttention gather**, đưa thẳng K/V mới sinh vào SDPA; KV write qua **C++ HIP kernel
  `reshape_and_cache`** thay indexing Python; attention chạy **FP16 SDPA với softmax acc FP32**.
  Kết quả đo công khai: 50.3 tok/s decode Qwen3.5-35B-A3B Q4_K_M (hơn Ollama 14%).
- Scheduler vLLM có sẵn **chunked prefill** (xem mục E).

### D. AMD-Ecosystem/llama.cpp — 6 PR set chính thức của AMD đang upstream (2026)
Toàn bộ là AMD làm cho ROCm backend (không CUDA). Set 2 (MMQ/MoE) đáng học nhất:
- **#39**: MoE `MUL_MAT_ID` — chọn **tile width theo ~2× số token/expert trung bình** thay vì worst
  case → **+91% pp128, +39% pp1024** trên Qwen3-35B-A3B.
- **#63**: **compacted MoE tiling** — chỉ launch số cột-block mà phân phối routing thật yêu cầu
  (mô phỏng `moe_align_block_size` của vLLM); mặc định bật.
- **#26301**: **mmvdq** — decode-path dequant-thẳng-float, bỏ qua bước quant q8_1 activation khi
  bandwidth-bound (thuần overhead); kèm fused gate+up SwiGLU.
- **#25**: **bảng MMQ theo arch**: `mmq_y` 128→64, warps/block 8→4 cho đúng arch đó → +14–18% pp.
  (Họ tune gfx1151, nhưng **thuật toán arch-hạc** — áp dụng gfx906 bằng cách tự quét lại 2 hằng số.)
- Set 5 = "GEMM Prefill Weight Optimizations" (đọc tiếp khi upstream).
- Tooling: set 1 có rocprof profiling hooks — dùng được ngay cho N5.

### E. Thuật toán tầng lịch (paper, không phụ thuộc vendor)
- **Chunked prefill — Sarathi-Serve (OSDI 2024)**; survey roofline: arXiv 2402.16363. Chunks cố
  định ~512–2048 token, xen kẽ decode để lấp "compute slack" của decode. Trên MI50, chunk là **biến
  quyết định trần byte** (bảng PCIe-vs-FLOP trong docs Strata) — Strata đã có chunking; vấn đề còn
  lại là **chính sách chọn cỡ chunk** (= N4).
- **Prefix/prompt caching** (không tính lại tiền tố): Strata có PROMPT_CACHE_TAIL/ORCA — trên card
  compute-bound, *đọc cache luôn thắng recompute*; đây là lever prefill "miễn phí" lớn nhất về TTFT.
- **KV quantization** (q8_0/q4_0) — trên gfx906 không mua FLOP attention (<1% FLOP) nhưng mua VRAM
  để đổi lấy expert cache — chỉ tối ưu theo thứ tự đó.

### F. ISA-level studies cho gfx906 (wiki-gfx906/skyne98 — đào trực tiếp trên 4× MI50/MI60, 60CU @1725MHz)
Đây là bộ số quý nhất trên mạng cho đúng kiến trúc này:
- **dot4/dot8 exploration**: codegen thật `__builtin_amdgcn_{s,u}dot4 → v_dot4_i32_i8/v_dot4_u32_u8`,
  `dot8 → v_dot8_i32_i4/u4`; đo **dot4 ~22 TOPS (chained) / ~44 TOPS (ILP-4)**;
  **dot8 ~44 / ~86 TOPS**; clamp/saturation hoạt động, acc 32-bit có overflow (wrap vs saturate).
  Công thức đỉnh: `CU×64 lanes×MACs×clock` → dot4 53 TOPS, dot8 106 TOPS.
- **Latency-hiding ops**: `v_mov_b32_dpp` 1.78 T xchg/s (~2× LDS exchange); `ds_bpermute_b32` 0.96 T;
  LDS: `b32` ~2–3.9 TB/s, `b64` ~4.3–8.8, **`b128` ~9.5–11.2 TB/s**; global `dwordx4` > `dword`;
  compiler tự xếp `s_waitcnt vmcnt(3..0)` staged — **cơ chế giấu latency chủ lực**; không có
  `s_clause`/`s_waitcnt_depctr`/`s_delay_alu` → ILP + occupancy là hai công cụ duy nhất.
- **FP32 vs QDQ dot**, **LDS layout standard for LLM blocks**, **KV-cache read/write study** — các
  study còn lại cùng loạt (trang wiki, chưa trích hết ở đây).

### G. Những đường bị loại (CUDA hoặc không tồn tại trên gfx906)
| Đường | Lý do loại |
|---|---|
| cuBLAS / cuDNN / CUDA fused int8 / `__dp4a` / `mma.sync` / `ldmatrix` / cp.async / Blackwell FP4-MMA | CUDA, loại theo yêu cầu và theo phần cứng |
| rocBLAS/CK **MFMA kernels**, AITer, Triton `tl.dot` | gfx906 không có `v_mfma` → crash ILLEGAL_INSTRUCTION (đo bởi cả 3 fork vLLM) |
| WMMA (bf16 HCD, `STRATA_WMMA_GEMM`, RDNA2 `rdna2_sgemm`) | RDNA/GCN5+1 không áp dụng cho gfx906 |
| vLLM quant kernels (AWQ/GPTQ Marlin, fp8) | dựa CUTLASS/Triton → không gfx906; GGUF qua llama.cpp path là cửa duy nhất |

---

## 3. ĐỐI CHIẾU VỚI NHỮNG GÌ Strata-gfx906-opt ĐÃ CÓ

Docs của dự án (`GFX906_PREFILL_METHODS.md` + `GFX906_MI50_REFIT_PLAN.md`) đã tự tìm ra phần lớn
mục A/E/F bằng đo: đường mặc định A = dequant FP16 + rocBLAS (93–99% đỉnh), N1 bucket+tile table
(trùng tinh thần PR #39/#63 của AMD), N2 fused-dequant-vào-LDS, N4 chunk theo trần byte, N5 wave64
butterfly. Hướng int8 MMQ bị đóng vì `udot4` **bị compiler ROCm 6.3.3 hạ xuống `v_macx_u8/v_madak_u8`**
→ chậm hơn FP16 3.8–12.4×.

**Mâu thuẫn mở quan trọng nhất giữa các nguồn:**
- Strata đo: "không V_DOT; int8 chạy ngang nhịp FP32".
- skyne98 (ROCm mới hơn, compile `-target amdgcn -mcpu=gfx906`, ISA dump xác nhận): **gfx906 có
  `v_dot4/v_dot8` thật, chạy ~86 TOPS (dot8, ILP-4)** — gấp ~6.5× ống FMA32 (~13.2 TFLOPS = 6.6
  TMAC/s).
Cả hai đều là đo thật; khác nhau ở **toolchain**, không phải silicon (trang assembler LLVM gfx906
liệt kê `v_dot*`; docs AMD 2018 gọi là DLOPS). ⇒ Kết luận tạm, trung thực: **hướng int8/int4 chưa
thể đóng cho tới khi microbench trên đúng toolchain dựng kernel Strata phân xử được `v_dot4` có
xuống ISA hay không.** Đây là đề xuất N7 bên dưới.

---

## 4. ĐỀ XUẤT THUẬT TOÁN (N1–N5 đã có trong docs → bổ sung N6–N9, mới từ cấu trúc card)

### N6 — **GEMM dot-int thủ công bằng inline asm (`v_dot8_i32_i4` / `v_dot4_i32_i8`)** — ✅ PROTOTYPE CHẠY ĐÚNG 21/10/2026
**Kết quả (xem [bench/dot-probe/RESULTS-N6.md](../bench/dot-probe/RESULTS-N6.md)):** kernel tay
wave64 (micro 4×4, LDS b128 swizzle, k-tile 128, prefetch, zero-spill, 256 `v_dot4_i32_i8`/k-tile):
gate/up-M40 **1.44×** hipBLAS HGEMM f16, M512 1.02×, down-M512 1.12× (9.2–9.5 TF/s); correctness 1.7e-6.
Đường int8 đã thắng rocBLAS f16 trên shape prefill thật + half VRAM weight + bỏ dequant pass.
Còn ~3× room tuning so với trần dot4 ILP4 (31 TF/s).
Nếu microbench xác nhận `v_dot` xuống ISA như skyne98:
- **Weight đã là 4-bit pair (IQ/IQ4/Q4-family) → `v_dot8` trực tiếp**: 1 lệnh = 8 MAC 4-bit.
- Act quant sang q8_1 (1 byte) → **`v_dot4`**: weight 8-bit × act 8-bit, 1 lệnh = 4 MAC.
- Trần lý thuyết: dot4-ILP4 ≈ 22 TMAC/s ≈ **3.3× FMA32**; dot8 ≈ **6.5×**. Ngay cả khi unpack/
  dequant ăn 50% VALU, vẫn còn 1.6–3× — lần đầu tiên prefill MI50 vượt trần 7.4 TFLOPS nội bộ.
- Thiết kế wave64: mỗi lane 1 qword weight + dword act; 4 accumulator độc lập (ILP4 — chính là
  pattern 22→44 TOPS); scale/partial-sum của IQ3_S xử lý bằng epilogue FMA (nhóm dot con rồi
  `v_mac_f32` với scale) — thứ tự cộng cố định để bit-parity có điều kiện.
- **Bắt buộc đi kèm N7 trước khi viết kernel đầy đủ.**

### N7 — `gfx906-dot-probe` — ✅ ĐÃ CHẠY 21/10/2026, PHÁN XỬ: dot-int CÓ THẬT, HƯỚNG MỞ
Kết quả đầy đủ: [bench/dot-probe/RESULTS.md](../bench/dot-probe/RESULTS.md). Tóm tắt:
đúng compiler build-906 (ROCm 6.3.3/clang 18) — `__builtin_amdgcn_udot4` hạ thẳng thành
**`v_dot4_u32_u8`** (ISA dump, zero `v_macx_u8`), `sdot8` thành **`v_dot8_i32_i4`**;
thông qua cùng nhịp phát lệnh với `v_fma_f32`, cho **3.9× (dot4) / 7.7× (dot8)** MAC-throughput
đo trên card đang bị service chia. Kết luận cũ "int8 đóng" (docs IQ3_S 3b) bị phản bác —
đó là hành vi của MMQ cũ, không phải giới hạn silicon. N6 dưới đây được mở chính thức.

*(Bản gốc N7 trước khi chạy — để lại làm ghi chú phương pháp:)*
Một TU HIP nhỏ: 4 kernel (a) `__builtin_amdgcn_udot4` thường, (b) udot4 inline asm
`v_dot4_i32_i8`, (c) sdot8/udot8 asm, (d) fma32 baseline. `hipcc`/`clang` của **đúng build-906**,
ISA dump bằng `llvm-objdump` + `rocprof` đếm instr. Nếu (a) ra `v_macx` còn (b) ra `v_dot4` →
Strata chỉ cần chuyển sang asm path. Nếu cả hai ra `v_macx` → LLVM của ROCm 6.3.3 không phát hành
dot trên gfx906, khi đó (và chỉ khi đó) dựng TU này bằng ROCm 7.x gfx906 (TheRock/ML-gfx906 toolkit,
chạy trong Docker theo đúng quy tắc không đụng /opt/rocm) rồi đo tiếp.

### N8 — Re-quant **phiên**: giải nén IQ3_S → **Q8_0 ngay trong staged copy host→device** (đề xuất mới) — ✅ ĐÃ CHẠY 21/10/2026, BIT-EXACT 100%
[bench/dot-probe/RESULTS-N8.md](../bench/dot-probe/RESULTS-N8.md): grid IQ3_S là số nguyên 0..15
⇒ re-quant sang Q8_0 **chính xác tuyệt đối** (1,048,576/1,048,576 bit-exact vs decoder ggml chuẩn),
0.034 ms/expert. Không mất mát, vô hình so với GEMM. N8+N6 thành pipeline thay đường f16.
Nút byte MoE = PCIe, nhưng nếu N6 thắng thì GEMM nhanh 3× mà byte giữ nguyên 0.43 B/w. Tận dụng
đường copy đã có: host thread (AVX2, CPU rảnh khi GPU quay) dequant IQ3_S→Q8_0 *trước khi đẩy qua
PCIe*, device nhận thẳng layout `v_dot4`-friendly (không unpack trên device, VALU device rảnh cho
FMA). Đánh đổi: PCIe byte ×2.4 → chỉ dùng được khi chunk ≥ cỡ mà **trần byte sau nhân 2.4 vẫn trên
trần dot-FLOP**; với 13.7 GB/s và 55.4 GiB phải tính lại bảng chunk (dạng đóng của N4), và expert
thường trú **không** cần phiên này (giữ nén trong VRAM, unpack 1 lần tại LDS = N2). Đây là thuật
toán lai byte/FLOP đầu tiên suy thẳng từ bất đẳng thức `dot_peak > FMA_peak` — chưa repo nào làm.

### N9 — Bảng tile MMQ riêng gfx906 quét bằng công cụ của AMD PR #25/#62 (đề xuất mới, rẻ)
AMD chứng minh chỉ 2 hằng số (`mmq_y`, warps/block) đổi theo arch ăn +14–18%. Strata đã có
`mmq-config-vega.cuh` (20-tiling) — nâng thành **cuối 2D nhỏ** (`mmq_y` ∈ {32,64,96,128} ×
LDS-buffer {1,2} × K-split) chạy `benchmark-moe-mmq` (chính PR #62 có generator phân phối routing
uniform/zipf — port sang `mmq_expert_bench`), xuất bảng tra theo shape Qwen3.8. Chi phí: một ngày
quét + một bảng hằng số; không đổi kernel.

### N10 — "Staged waitcnt" tự điển hình hoá (bổ trợ N2)
Vì không cp.async, kernel LDS nào cũng phải tự pipeline. Chuẩn hoá thành macro:
`issue_loads(buf[k+1]); s_waitcnt vmcnt(N); ds_write_b128 buf[k+1]; compute(buf[k]); s_waitcnt lgkmt(N)`
đúng pattern ILP mà compiler tự xếp khi đo (mục 2F) + `dwordx4` cho global và `b128` cho LDS.
Đây chính là cơ chế ẩn sau "max-ilp +2–8%" của bản Strata trong ML-gfx906.

*(N1–N5 trong `docs/GFX906_PREFILL_METHODS.md` mục 6 vẫn đúng thứ tự ưu tiên; N2 vẫn là ứng viên số
một nếu N7 phân xử rằng dot-int không xuống ISA.)*

---

## 5. NGUỒN (link chính)

**Kiến trúc gfx906 / ISA**
- skyne98 wiki-gfx906 (bộ study 4×MI50/MI60): `mi50-mi60-architecture-baseline`,
  `gfx906-dot4-dot8-exploration`, `gfx906-latency-hiding-ops`, `special-isa-quant-dequant`,
  `lds-layout-standard`, `fp32-vs-qdq-dot` — https://skyne98.github.io/wiki-gfx906/
- LLVM assembler gfx906 (liệt kê `v_dot*`, không `v_mfma*`): https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html
- LLVM gfx908 (đối chứng: `v_mfma*` chỉ từ đây): https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX908.html
- AMD press release MI50/MI60 (26.8/13.4 TFLOPS, "DLOPS"): ir.amd.com 06/11/2018
- Vega 7nm Shader ISA PDF: gpuopen.com

**llama.cpp / HIP**
- AMD-Ecosystem PR sets (6 discussion #26333/#26349/#26377/#26378/#26379/#26380; #39, #63, #26301, #25, #62)
- iacopPBK/llama.cpp-gfx906 (→ mxxm-t/mx-llama.cpp); eslowney; fuutott; bearqq (level1techs thread 254257)
- ML-gfx906 (mixa3607) releases — có build Strata với `max-ilp`
- Sebastian Haas case study "the kernel that waited for itself" (db-gemm, LDS ping-pong, split-K,
  occupancy 3 waves/SIMD) — https://sebastianhaas.dev/research/case-studies/m12-db-gemm/

**vLLM gfx906**
- kxb520/vllm-mi50, noshitcoding/vllm-mi50 (Direct K/V Prefill, ROCM_ATTN redirect, Tensile hsaco),
  btbtyler09/vllm-rocm-gcn5, kausik/vllm-gfx906

**Tầng lịch / lý thuyết**
- Sarathi-Serve chunked prefill (OSDI'24); LLM Inference Survey + Roofline: arXiv:2402.16363
- ACM SIGOPS "Efficient LLM Inference via Chunked Prefills" (2025)
- rocBLAS/Tensile docs (arch-specific GEMM tuning, TensileLibrary): rocm.docs.amd.com

**Trong máy**
- `docs/GFX906_PREFILL_METHODS.md`, `GFX906_MI50_REFIT_PLAN.md`, `GFX906_SINGLE_MI50_IQ3_S.md`
  (mọi con số "đo" trong bài này lấy từ đó, kèm ngày/máy).
