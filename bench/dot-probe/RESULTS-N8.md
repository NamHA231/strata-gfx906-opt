# N8 — IQ3_S→Q8_0 exact requant: KẾT QUẢ (đo 21/10/2026)

File: `n8_requant.hip` (`./n8`), grid table trích từ `ggml-common.h` của llamacpp trong repo (`iq3s_grid.h`).

## Phát hiện thuật toán (mới, từ cấu trúc IQ3_S + N7)

IQ3_S = `d·(1+2·nibble4bit)·(±grid[idx9bit])`, grid magnitudes là **số nguyên 0..15**
(max đo = 15, xác minh từ bảng 512-entry của ggml). ⇒ re-quant IQ3_S→Q8_0 (int8 + scale fp32/32)
**CHÍNH XÁC TUYỆT ĐỐI, không mất mát**: int8 = ±grid, scale = d·(1+2·nib). 1048576/1048576 phần tử
bit-exact với decoder chuẩn ggml. Không cần "chấp nhận sai số re-quant" như lo ngại ban đầu.

## Số đo

| shape | thời gian | traffic |
|---|---|---|
| K=1280 N=2560 (gate/up 1 expert) | 0.034 ms | đọc 1.41 MB IQ3_S (41 GB/s), ghi 3.28 MB Q8_0 (96 GB/s) |
| K=2560 N=1280 (down) | 0.033 ms | tương tự |

Kernel thô 1 thread/256-block (chưa tune) đã chỉ ≈ 9% thời gian một lần GEMM n6q cùng shape —
và chạy MỘT LẦN per weight-set, không phải per token.

## Pipeline N8+N6 chốt cho Strata (thay đường f16 hiện tại)

1. Host giữ GGUF IQ3_S (3.44 bpw) → staged copy PCIe chỉ 1.4 MB/expert (bằng 1/2 f16).
2. Kernel `iq3_to_q8` (đã có, ~0.03 ms/expert) bung thành int8 + fp32 scale/32 trong VRAM.
3. `n6q` dot4 GEMM ăn trực tiếp (cùng layout scale [N][K/32] fp32) — không dequant f16 pass.
4. VRAM expert: 1 byte/weight + 1/32 float ≈ half so với f16 hiện tại → nhét được nhiều expert
   resident hơn, giảm thrash khi rotate 64 experts.

## Số đo end-to-end trên service thật (restart 21/10/2026, model coder IQ3_S 27B, port 8080)
- Baseline (env off, binary mới): pp = 207.7–210.0, median **209.3 tok/s** (5 lần, prompt ~2.9k token unique).
- `STRATA_GFX906_DOT4=1`: pp = 214.1–223.4, median **215.1 tok/s** → **+2.8%**.
- Lưu ý: request ĐẦU TIÊN sau restart có thể rất chậm/timeout — code object HIP nạp ở lần gọi đầu; số nào cũng loại warm-up.
- +2.8% end-to-end trong khi GEMM expert nhanh 1.3–1.6× → expert GEMM chỉ là MỘT PHẦN của pp; service đang chạy `STRATA_EXP_MODE=8` (fused kernel moe_fused*, có thể đi đường khác Gemm::native) — hướng nâng tiếp theo là đưa dot4 vào chính đường fused mode 8.


## Việc còn của bước integration

- Requant chạy trên stream, overlap với GEMM expert trước (double-buffer slot).
- Act quant f32→q8_0 (per-32, ~vài µs) sát GEMM; đã có pattern trong N7 probe.
- Tune kernel requant (vectorize 4 super-block/thread) nếu cần, nhưng 0.03 ms/expert đã vô hình.
