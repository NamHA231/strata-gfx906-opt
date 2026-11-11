# N6 — GEMM dot4 prototype: KẾT QUẢ (đo 21/10/2026)

**VÔ ĐỊCH: `n6q.hip` / `./n6q` (k-tile 128).** k-tile 64 (`n6p`) chỉ đạt 0.56–1.01× f16;
gấp đôi độ sâu k-tile (sync thưa hơn 2×, prefetch 2 quad) nâng thẳng lên **0.84–1.44× f16,
9.2–9.5 TF/s @M512** — scheduling là trần thật, không phải LDS hay ISA.

| shape (K×N) | n6q dot4 | hipBLAS f16 | dot4/f16 |
|---|---|---|---|
| gate/up M40 K=1280 | 0.067 ms (3.89 TF/s) | 0.097 | **1.44×** |
| gate/up M128 | 0.127 (6.63) | 0.152 | 1.20× |
| gate/up M512 | 0.363 (9.24) | 0.372 | 1.02× |
| down M40 K=2560 | 0.123 | 0.103 | 0.84× |
| down M512 | 0.354 (9.48) | 0.396 | **1.12×** |

Correctness max rel err 1.7e-6 (i32 ref CPU). VGPR ~zero-spill với launch_bounds(256,1).

## Kết luận N6 (thay thế mọi số cũ bên dưới)

1. GEMM int8×int8 bằng `v_dot4_i32_i8` kernel tay **ĐÃ THẮNG rocBLAS f16** trên shape prefill thật
   (1.44× với M=40 — đúng shape expert MoE; 1.02–1.12× với M=512 — đúng shape dense/attention).
   Kèm 2 bonus cấu trúc: expert weight ở 1 byte/weight (half VRAM, half PCIe traffic so với f16
   path của Strata) và không cần dequant pass.
2. 9.5 TF/s = ~30% trần dot4 ILP4 (probe 31.2): room tuning Tensile-style còn ~3× — giao phần
   tinh chỉnh tile/LDS cho bước integration vào engine.
3. down-M40 0.84×: utilization (20 blocks/60 CU). MoE thật batch 64 experts → 1280 blocks, hết problem.
4. Công thức kernel chốt cho Strata: block 256, tile 64×64, k-tile 128, swizzle `(row&7)^quad`,
   micro 4×4/lane, i32 flush per 32, prefetch quad k+128/k+192, launch_bounds(256,1).

## Số đo tham chiếu — các vòng lặp trung gian (đã vượt)

| phiên bản | gate/up M40 | gate/up M512 | down M512 |
|---|---|---|---|
| n6 (64-tile, 39 VGPR spill) | 0.51× | 0.42× | 0.44× |
| n6b (zero spill) | 0.90× | 0.81× | 0.79× |
| n6p (+prefetch) | 1.01× | 0.83× | 0.68× |
| **n6q (k-tile 128)** | **1.44×** | **1.02×** | **1.12×** |

Design: block 256=4 waves × wave-tile 32×32 (lane micro 4×4 = 16 acc), LDS k-tile swizzle chống
conflict (`(quad ^ row&7)<<2`, LDS b128 trong/ra), q8_0 scale-block-32, i32 acc flush mỗi 32 element.
Đúng compiler build-906. Đối thủ: hipBLAS HGEMM f16→f32 (đường Strata native, Tensile ~67–72% đỉnh f16).

## Số đo (best-of-3, service vẫn chạy chia card)

| shape (K×N) | dot4 n6p | hipBLAS f16 | dot4/f16 |
|---|---|---|---|
| gate/up M40 K=1280 | 0.097 ms | 0.098 | **1.01×** |
| gate/up M128 | 0.158 | 0.153 | 0.97× |
| gate/up M512 | 0.448 (7.49 TF/s) | 0.370 (9.06) | 0.83× |
| down M40 K=2560 | 0.182 | 0.101 | 0.56× |
| down M512 | 0.584 (5.75 TF/s) | 0.399 (8.41) | 0.68× |

Correctness: max rel err 1.7e-6 vs reference i32 chính xác trên CPU (64×64×128).

## Diễn giải

1. **Kernel tay dot4 đạt 0.56–1.01× đường rocBLAS f16** — GEMM int8 không còn là giấc mơ trên MI50,
   nhưng prototype v1 chưa thắng rõ. Giá trị thắng thực sự của đường int8 nằm ở CHỖ KHÁC:
   không trả phí dequant pass (đọc 2× byte + ghi f16), weight nằm VRAM 1 byte/weight (half expert set),
   và expert GEMM chạy thẳng từ GGUF block.
2. down-M40 0.56× là **bài toán utilization không phải algorithm**: grid = N/64 blocks;
   N=1280 → 20 blocks trên 60 CU (1.3 wave/CU, latency lộ liễu). Với MoE thật, shape này là M40
   × 64 experts theo batch → 1280 blocks, problem biến mất. Gate/up N=2560 (40 blocks) đã 1.01×.
3. Khoảng cách với trần dot4 (ILP4 probe: 31.2 TF/s; GEMM: 7.5 = 24%): không phải LDS bandwidth
   (LDS:dot = 36:256 instr, b128 hết), không phải spill — mà là **scheduling**: syncthreads/k-tile,
   LDS latency không được che bằng s_waitcnt staging. Đó đúng là chỗ N10 (latency-hiding pattern
   của skyne98) nhắm vào; double-buffer LDS + prefetch (đã làm ở n6p) chỉ mang lại +5–15%.
4. Con số để neo kế hoạch: **GEMM dot4 thực dụng ~7.5 TF/s hôm nay** so với f16 Tensile 9.1 —
   hòa vốn về FLOP, thắng về memory. Nhân thêm khi scheduling tốt dần.

## Bước tiếp (thứ tự đã kiểm chứng độ hứa hẹn)

- 2-stage LDS double buffer + s_waitcnt thủ công (N10 pattern) → mục tiêu ≥1.3× f16 shape lớn.
- Grid config theo M: tile N=32 cho N nhỏ/số block thiếu; split-K khi M nhỏ AND N nhỏ.
- Integrate vào Strata: thay `gemm_q4_k` path bằng kernel này + path N8 (re-quant IQ3→Q8 khi staged
  copy) để expert weight không bao giờ phải dequant sang f16.
