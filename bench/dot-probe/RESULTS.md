# N7 dot-probe — KẾT QUẢ & PHÁN XỬ (đo 21/10/2026)

Toolchain: đúng compiler của `build-906` — `/opt/rocm-6.3.3` (clang 18, roc-6.3.3), `--offload-arch=gfx906`.
Máy: MI50 32 GB, **Strata service vẫn đang chạy** (toàn bộ số dưới đây đo lúc card bị chia — tỷ số là thứ đáng tin, giá trị tuyệt đối là cận dưới).
Geometry: GRID=2048, BLK=256, ITERS=65536, best-of-3 (pattern Benchmark A + ILP4 của wiki-gfx906).
FLOP đếm: dot4=8, dot8=16 mỗi lệnh ×1 (chained) / ×4 (ilp4); fma=2.

## ISA dump (roc-obj-extract + llvm-objdump --mcpu=gfx906)

| kernel | lệnh phát hiện |
|---|---|
| `k_dot4_builtin_chained` / `_ilp4` | **v_dot4_u32_u8** ×10 / ×9 — KHÔNG có v_macx_u8/v_madak_u8 |
| `k_dot4_asm_chained` / `_ilp4` | v_dot4_u32_u8 (asm buộc, hoạt động) |
| `k_dot8_builtin_chained` / `_ilp4` | **v_dot8_i32_i4** ×5 / ×4 |
| `k_dot8_asm_chained` / `_ilp4` | v_dot8_i32_i4 (assembler gfx906 của ROCm 6.3.3 CHẤP NHẬN) |
| `k_fma_*` | v_fma_f32 (baseline) |

## Throughput đo được

| kernel | ms | TFLOP/s |
|---|---|---|
| fma32 chained / ilp4 | 15.6 / 34.2 | 4.4 / 8.0 |
| dot4 builtin chained / ilp4 | 15.2 / 35.3 | 18.1 / 31.2 |
| dot4 asm chained / ilp4 | 16.2 / 35.4 | 17.0 / 31.0 |
| dot8 builtin chained / ilp4 | 15.1 / 35.4 | 36.3 / 62.1 |
| dot8 asm chained / ilp4 | 16.1 / 35.5 | 34.2 / 62.0 |

## Phán xử cho N6/N8 (mở lại hướng int8/int4)

1. **`v_dot4_u32_u8` và `v_dot8_i32_i4` tồn tại, chạy đúng, và compiler 6.3.3 tự phát ra từ builtin.**
   Kết luận "int8 đóng vì udot4 hạ xuống v_macx_u8" (docs IQ3_S 3b) **bị phản bác** — đó là hành vi của
   code hình thức MMQ cũ, không phải giới hạn silicon. Mâu thuẫn với wiki-gfx906 được giải quyết theo
   hướng wiki-gfx906 ĐÚNG.
2. `v_dot*` tiêu thụ **cùng slot phát lệnh** với `v_fma_f32` (ms gần bằng nhau ở cùng số lệnh):
   dot4 = **3.9×** MAC, dot8 = **7.7×** MAC so với ống FP32 trên cùng card, ngay cả khi card đang bị
   service chia. ILP4 nhân ~1.7–2× so với chained — accumulator độc lập là bắt buộc.
3. builtin ≡ asm về ISA và tốc độ ⇒ kernel mới chỉ cần builtin, không cần inline asm
   (asm vẫn giữ làm保険 đường nếu về sau compiler regress).
4. Hệ quả pipeline: trần FLOP mới cho GEMM expert = **~26 TMAC/s dot4** (dot8 4-bit: ~52 TMAC/s),
   gấp 3.5–7× trần FMA 7.4 TFLOPS mà toàn bộ kế hoạch N1–N5 đang neo vào. N6 (GEMM dot-int)
   **CHÍNH THỨC được mở**; N8 (re-quant phiên để không nhân byte PCIe) theo sau.

## Tồn tại của phép đo (theo quy tắc số của repo)

- Đo lúc service đang chạy → giá trị tuyệt đối là cận dưới; tỷ số dot/fma ít bị ảnh hưởng
  (cùng occupancy hình thức, cùng lịch phát lệnh).
- Chưa đo: tiêu thụ điện/nhiệt khi dot bão hoà 60 CU lúc card rảnh; chưa đo tính ĐÚNG CỦA GEMM
  thật (dot chỉ là nguyên liệu — cần kernel LDS 2D thật mới kết luận cho prefill).
- Chạy lại lúc card trống: `./probe4 && ./probe8` (binary đã build ở đây).
