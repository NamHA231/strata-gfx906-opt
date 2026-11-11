# Prefill trên AMD MI50 (gfx906): mọi đường chạy được, và cốt lỗi suy từ kiến trúc card

Tài liệu này liệt kê **toàn bộ** cách làm prefill có trong mã nguồn Strata, chỉ ra cách nào thật sự
chạy trên một Instinct MI50, và đi tới cốt lỗi bằng cách suy từ chính microarchitecture của gfx906
chứ không suy từ cách các mô hình khác đang làm. Kế hoạch hành động rút ra từ đây nằm ở
[GFX906_MI50_REFIT_PLAN.md](GFX906_MI50_REFIT_PLAN.md).

Phạm vi: **một** card gfx906 (MI50 32 GB là máy đo chính; MI50 16 GB và Radeon Pro VII được ghi rõ
khi số lấy từ đó), pack **IQ3_S** (`--native`), ROCm 6.3.3, `build-906`. **Không có đường nào phụ
thuộc CUDA ở trong tài liệu này** — mục 2 ghi rõ những đường nào bị loại và vì sao chúng không tồn
tại trên card này.

Quy tắc số: mỗi con số hoặc là **đo** (kèm máy, ngày, lệnh), hoặc là **phép tính từ đầu vào đã đo**
(ghi rõ là phép tính), hoặc là **chưa đo** (ghi rõ là chưa đo). Theo [AGENTS.md](../AGENTS.md).

---

## 1. Cốt lỗi, nói trước

Prefill trên MI50 bị buộc bởi **nhịp `v_fma_f32`**, không phải băng thông HBM, không phải độ dài
attention, không phải công thức nhân ma trận. Ba hệ quả của cấu trúc card:

1. **Không có khối gia tốc nào.** gfx906 không có WMMA, không MFMA, không tensor core, không
   `V_DOT`/`v_dot4_i32_i8`, không SIMD byte đóng gói. Mọi nhân–cộng cho matmul đi qua cùng một cổng
   FMA FP32. `__builtin_amdgcn_udot4` xuống thành chuỗi `v_macx_u8`/`v_madak_u8`: **int8 chạy ngang
   nhịp FP32**. `v_pk_fma_f16` cũng vậy. ⇒ *trên card này, lượng tử hoá mua byte, không bao giờ mua
   FLOP* ([GFX906_SINGLE_MI50_IQ3_S.md](GFX906_SINGLE_MI50_IQ3_S.md) mục 3b). Đã được xác nhận bằng
   đo: đường int8 MMQ chậm hơn đường FP16 **3.8–12.4× ở mọi `T`** (mục 6b của tài liệu đó).
2. **Trần FLOP là ~7.4 TFLOPS FP32.** Với ~12 GFLOP/token (mục 3b), trần tuyệt đối là **~617 tok/s**
   ở mọi shape, mọi kernel. Số đo end-to-end hiện tại 330–350 tok/s ⇒ 4.2–4.6 TFLOPS hiệu dụng ⇒
   **57–62% đỉnh**. GEMM FP16 cô lập đã đạt **6.86–7.35 TFLOPS = 93–99% đỉnh** tại `T ≥ 512`
   (đo 19/10/2026, một MI50 32 GB, median of 20, shape tổng hợp). Room còn lại của prefill là
   **1.5–1.7×**, và nó nằm ở **thời gian ngoài GEMM**.
3. **Trần byte cho card 32 GB là PCIe, và cỡ chunk là biến quyết định nó** (mục 5). Ở
   `--prefill 2048` hai trần này nằm *gần bằng nhau*, nên không một cải thiện kernel đơn độc nào đưa
   tới 600 tok/s.

Câu trả lời ngắn cho "cốt lỗi prefill dựa vào cái gì": **dựa vào số FMA mà 60 CU kịp phát trong một
giây**, và mọi thứ khác (byte, attention, kernel launch) chỉ là thứ được phép đứng dưới con số đó.
Vì GEMM đã ở 93–99% đỉnh, việc còn lại là **làm cho toàn bộ thời gian khác biến mất hoặc chồng lấp**,
không phải viết GEMM mới.

---

## 2. MI50 cho gì và bắt gì — bảng suy luận

| Đặc tính gfx906 | Hệ quả cho prefill |
|---|---|
| 60 CU, wave64 (không warp32) | Tile phải là bội của 64 lane. Cây giảm 32 lane mà lớp HIP-compat sinh ra tốn ~5 lần shuffle thay vì 1 butterfly 64 lane; `__syncwarp` = fence + `wave_barrier` + fence trên cả 64 lane. Đó là lý do `reduce12_gfx906` tồn tại. |
| Không WMMA / MFMA / tensor core | `dev_info()` ở [moe_fused_iq.cu:1258](../src/prefill/moe_fused_iq.cu#L1258) trả `DevInfo` rỗng vì gfx906 báo compute capability 9.0 ⇒ mọi kernel `mma.sync` của đường prompt **không bao giờ được gọi**. Attention prompt rơi về `qsa_decode_attn_batch` FP32 ([qsa_prompt_attn.cu:2262](../src/kernels/cuda/qsa_prompt_attn.cu#L2262)). |
| Không `V_DOT` | int8 không được 4 byte/lane. Mọi kế hoạch "TOPS INT8" đều sai trên kiến trúc này (bảng phủ nhận, mục 6b tài liệu IQ3_S). |
| FMA hai thành phần, ~7.4 TFLOPS FP32; FP16 packed **cùng nhịp** | Trần matmul là 7.4 TFLOPS. Không có đường "nhân đôi bằng FP16" trên card này. |
| LDS 64 KiB/CU, **không có `cp.async`** | Mỗi load vào LDS là VMEM → register → LDS: nhạy latency hơn hẳn CDNA/RDNA3. Tile phải nhỏ và **đồng nhất**; GEMM grouped-ragged theo expert trả giá latency này nhiều hơn trên gfx906 so với card khác. |
| HBM2 ~1.0 TB/s, 32 GB | Chứa được dense side + KV + PLE, **không** chứa 55.4 GiB routed experts ⇒ prefill dựa vào copy engine host→device. |
|hipBLASLt tuned-algo table không áp dụng cho gfx906 | Đường dense dùng rocBLAS + solution table riêng (`STRATA_ROCBLAS_TUNING`). |

**Những đường bị loại hoàn toàn** (không phải "chậm", mà là không có trên kiến trúc này):

| Đường | Vì sao loại |
|---|---|
| `Gemm::bf16()`, `Gemm::bf16_hcd_exact()` | Cần WMMA (bf16 HCD là bit-repro của hipBLASLt algo 1176/1177, N320/K10240) — gfx11+ |
| `rdna2_sgemm()` | RDNA2 |
| `STRATA_WMMA_GEMM=1` | RDNA3 / RDNA3.5 ([gemm.cu:1062](../src/prefill/gemm.cu#L1062)) |
| Mọi kernel `mma.sync` / tensor-core của prompt path | Không có khối đó trên gfx906 (dòng `dev_info()` ở trên) |
| GDN chunked | Chỉ gfx11 ([AMD_HIP.md](AMD_HIP.md)) |
| cuBLAS / cuDNN / CUDA fused int8 | Loại theo yêu cầu; và chúng không tồn tại trên HIP |

---

## 3. Bản đồ prefill: các pha thật, lấy từ mã nguồn

`PfPhase` ở [prefill.cpp:2287](../src/prefill/prefill.cpp#L2287) là bản đồ thật của một chunk, và
`STRATA_PREFILL_TIMING=1` in ra thời gian cho từng pha. Đây vừa là danh sách thứ phải đo, vừa là
danh sách thứ *được phép* tồn tại ngoài GEMM:

```
embed+steps · hc read · gdn (proj / conv+gates / recurrence / out) · qsa proj · kv stage
qsa indexer · qsa select · qsa attn · router+shared · host grouping · gather · wait copy
dequant · gemm gate/up · gemm down · combine · ple
```

Một chunk đi qua hết 48 layer; state (GDN recurrence + conv, KV pool QSA + indexer, PLE history)
được để lại đúng chỗ đường token để lại ([prefill.hpp](../include/strata/prefill/prefill.hpp)).

---

## 4. Toàn bộ cách đang có, và cách nào chạy trên MI50

| # | Cách | Cơ chế | Trên MI50 | Số đo / trạng thái |
|---|---|---|---|---|
| A | **`Gemm::native()` = `dequant_f16` + rocBLAS FP16** | Weight GGUF → scratch FP16, rồi GEMM FP16-vào/FP32-ra ([gemm.cu:1301](../src/prefill/gemm.cu#L1301)) | **Đường mặc định đang chạy**, cho dense và cho experts | GEMM cô lập 6.86–7.35 TFLOPS @ `T ≥ 512` = 93–99% đỉnh (đo 19/10, MI50 32 GB) |
| B | **`moe_mmq_gfx906` (int8, `udot4`)** | q8_1 activation × weight int8, giảm theo wave | Chạy **đúng** (rel L2 0.0037–0.0038, max abs 0.021–0.051, không NaN/Inf) nhưng **chậm** | Chậm hơn A **6.3× (T=128) tới 11.2× (T=512)** cho iq3_s gate/up, **3.8–7.5×** cho iq4_nl down. Default `STRATA_GFX906_MMQ_MAX_T` **512 → 0** ([gemm.cu:1273](../src/prefill/gemm.cu#L1273)). Hướng int8 **đóng** cho tới khi kernel có LDS tile 2D |
| C | **`f16_inplace()` / `STRATA_HIP_PROMPT_F16=1`** | GEMM FP16 vào **và ra**, widen tại chỗ ([gemm.cu:539](../src/prefill/gemm.cu#L539)) | Có hiệu lực, opt-in, **không** bit-identical với mặc định | gfx103x ~2×. **gfx906 chưa đo**; vì FP16 cùng nhịp FP32, ở đây chỉ tiết kiệm băng thông ghi `Y`, không tiết kiệm FLOP |
| D | **`STRATA_PF_GEMM=1` (+ `STRATA_PF_PAD=1`)** | GEMM tự viết `strata_pf_gemm_f16`, row stride `K+64` để stride bội 4 KB không camp kênh nhớ ([gemm.cu:1236](../src/prefill/gemm.cu#L1236)) | Biên dịch được trên HIP; **gfx906 chưa đo** | Ghi chú trong code trích 19→29 TFLOPS từ `gemm_probe8` — **đó là card khác**; trên MI50 trần đã là 7.4 TFLOPS nên PAD chỉ còn nghĩa khi đang dưới đỉnh |
| E | **Expert mode 8, `native_gu_fused_kernel`** | Fused gate+up + SwiGLU + q8_1 cho expert IQ | **Đã là default gfx906** (commit `10230425`) | mode 8 vs 7, **bitwise bằng nhau** (đo 18/10, MI50 32 GB): layer5 10×1 101.1→89.5 µs (**−11.5%**), 10×4 216.9→190.6 (**−12.1%**), 40×8 1383.8→1423.7 (**+2.9%**), layer0 4×8 170.0→162.3 |
| F | **Chunked prompt** (`--prefill auto` ≤ 8,192; 32,768 opt-in) | Chunk đi qua 48 layer, state giữ giữa chunk | Đang chạy; **biến tốc lớn nhất chưa được quét** | Xem mục 5 |
| G | **Expert ring + stager + stream-all** | `RING_MAX=1024`, `ring_bytes`/`ring_slots`/`stage_slots`, staged walk từ 1,024 token, `STRATA_PREFILL_RING`, `STRATA_PREFILL_STREAM_MIN`, 96 slot khi host copy ≥90% | Đang chạy. `STRATA_PREFILL_CPU_SHARE` **tắt trên AMD/HIP** | Chưa quét slot ring trên IQ3_S (mục 6b điểm 3 của tài liệu IQ3_S vẫn mở) |
| H | **Expert cache / `--resident-experts` / cho vay slot** | Expert thường trú VRAM; prompt mượn slot top rồi refill | Đang chạy | Hit 0.97–0.995 trên MI50 32 GB (báo cáo cộng đồng, **IQ2_XS**, 19,304 slot). Nếu hit rate IQ3_S < ~0.95 thì **nút thắt là cache, không phải kernel** |
| I | **QSA attention FP32 wave64** | `qsa_decode_attn_batch`; sparse width = `min(n_kv, idx_top_k + idx_block − 1)` = **tối đa 2,051 cell, không đổi theo độ dài prompt** | Đường **duy nhất** khả dụng trên gfx906 | ≈50 MFLOP/token (tính từ shape mặc định trong `geometry.hpp`, **chưa** đọc từ checkpoint thật) ⇒ **<1%** FLOP mỗi token. **Đừng tối ưu attention trước** |
| J | **`STRATA_SELECT_SIMT=1`** | Bộ chấm điểm QSA tiled FP32 | Trên gfx906 đây là đường *tự nhiên* (không có tensor core để né) | gfx103x +7.4% ở prompt 128K; **gfx906 chưa đo** |
| K | **GDN linear attention** | O(1)/token; recurrence đi trong một kernel theo chunk | Chạy trên gfx906 (bản chunked chỉ gfx11) | `gfx906_gdn_rec_head` parity pass |
| L | **PLE ở IQ4_NL thay vì Q8_0** | 26.8 GiB thay vì 50.7 GiB | Có hiệu lực | Trên card 32 GB, 24 GiB chênh mua thêm slot expert cache |
| M | Switch A/B nhỏ: `STRATA_PF_FUSED`, `STRATA_GFX906_ATTN_QUERY_SWIZZLE`, `STRATA_GFX906_ATTN_REDUCE12`, `STRATA_MMQ_KQ`/`STRATA_MMQ_KQUANTS`, `STRATA_HC_REQ8`, `STRATA_HC_UPMIX`, `STRATA_ROUTE_PER_TOKEN`, `STRATA_GR_FAST`, `STRATA_GR_SPLIT`, `STRATA_MMVQ_WAVE` | Xem [GFX906_SINGLE_MI50_IQ3_S.md](GFX906_SINGLE_MI50_IQ3_S.md) mục 4.3 | Đa số có hiệu lực trên gfx906 | Nhiều cái **chưa có số đo cô lập từng cái**; `STRATA_GR_SPLIT` không bit-exact nên off |

---

## 5. Hai trần, và phép tính đặt chunk

Đầu vào đã đo: routed experts IQ3_S **55.4 GiB** (bảng mục 3 tài liệu IQ3_S); PCIe host→device
**13.7–13.8 GB/s** (đo trên Radeon Pro VII 16 GB, **chưa đo lại trên chính MI50 32 GB**);
đỉnh FP32 **~7.4 TFLOPS**; ~12 GFLOP/token (mục 3b).

Với một chunk dài `C`, gần như mọi expert đều được route tới (10/512 mỗi token), nên mỗi chunk phải
kéo ~toàn bộ 55.4 GiB qua PCIe một lần. Phép tính từ đầu vào trên:

| chunk `C` | MiB experts / token | Trần tok/s do PCIe | Trần nào thắng |
|---|---|---|---|
| 1,024 | 57.1 | ~240 | **byte** — PCIe là nút thắt |
| 2,048 | 28.5 | ~480 | hai trần gần bằng nhau (đang ở đây) |
| 3,072 | 19.0 | ~720 | **FLOP** |
| 4,096 | 14.3 | ~960 | **FLOP** rõ ràng |

Đối chiếu độc lập: 330–350 tok/s × 28.5 MiB/token = 9.4–10.0 GB/s = **69–73% của 13.7 GB/s**, khớp
với việc prefill đo được ở 57–62% đỉnh FLOP. Hai con đường độc lập cho cùng một bức tranh — đó là lý
do bảng này đáng tin hơn từng con số lẻ.

Hệ quả, và là thứ dễ bị hiểu sai nhất:

- `--prefill 2048` là ràng buộc **dung lượng buffer**, không phải ràng buộc tốc độ. Trên card 32 GB
  nó đặt prefill đúng vào điểm giao của hai trần.
- Dưới `C ≈ 3,072`, làm kernel nhanh hơn không tăng tok/s đáng kể — bạn đang chờ PCIe.
- Trên **MI50 16 GB** hoặc Pro VII, 55.4 GiB experts không vào nổi, hit rate giảm, và cả prefill lẫn
  decode rơi về vùng byte. Dự án này đo trên 32 GB; kết luận cho 16 GB phải đo riêng.

---

## 6. Năm hướng thuật toán, mỗi cái xuất phát từ một đặc tính ở mục 2

Không sao chép từ các repo khác; mỗi hướng dưới đây là hệ quả của một dòng trong bảng mục 2.

### N1 — GEMM expert theo *bucket có đệm + bảng tile*, thay cho grouped-ragged
**Từ:** LDS 64 KiB + không `cp.async` + wave64.
Chunk 2,048 cho mỗi expert trung bình `10/512 × 2048 ≈ 40` token — rất xa điểm 93–99% đỉnh đo ở
`T ≥ 512`, và mỗi tile lệch kích thước trả giá latency VMEM→register→LDS. Thuật: counting-sort token
theo expert, **đệm danh sách token của mỗi expert lên bucket luỹ thừa hai** (16/32/64/128), chạy
**một launch mỗi bucket** với một bảng trỏ tile → `(expert, row-tile)`. Số launch mỗi layer xuống
~4–6 thay vì 512; phần đệm thừa trung bình nhỏ hơn nhiều so với đệm cứng một cỡ.
**Điều kiện chấp nhận:** parity bitwise qua `mmq_expert_bench` với GGUF thật — thứ tự cộng phải giữ
nguyên, nếu không thì phải là switch opt-in kèm KL.

### N2 — Bỏ pha `dequant`: lượng tử giải trực tiếp vào LDS tile
**Từ:** expert không dùng lại giữa các chunk, và LDS chứa đủ tile.
Hiện mỗi GEMM đọc weight nén (~0.43 B/weight ở IQ3_S), **viết** 13.1 MB FP16 cho một expert gate/up
1280×2560, rồi GEMM **đọc lại** đúng 13.1 MB đó: ba lượt qua HBM thay vì một. LDS 64 KiB/CU chứa đủ
tile FP16 32×32 (2 KiB) nhân bốn bộ đệm. Thu được: gần như toàn bộ pha `dequant` trong `PfTimer`,
cộng với phần `gemm` vì weight chỉ đọc một lần và ở dạng nén.
Đây là ứng viên số 1 mà mục 6b của tài liệu IQ3_S đã chỉ tên nhưng chưa ai viết.

### N3 — Cache FP16 đã giải nén cho expert **thường trú**
Ngược hướng với N2, và chỉ đúng cho expert resident: giữ bản FP16 trong VRAM để đổi 4.6× byte lấy
0 lần dequant. Chỉ hợp lý trên gfx906 vì byte HBM không phải nút thắt của phần card đầy. Phải đo
theo hit rate thật; trên 16 GB hướng này gần như chắc chắn sai.

### N4 — Chọn chunk theo trần byte, không theo bộ nhớ còn lại
Đặt `C` = giá trị nhỏ nhất sao cho `55.4 GiB / C < 0.85 × PCIe_đo_được`, để FLOP là trần. Trên máy
đo chính: **C ≥ 3,072**. Đây là thay đổi chính sách lập kế hoạch (mục 5), không phải kernel.

### N5 — Quy ước wave64 cho mọi reduction trong prompt path
**Từ:** lớp HIP-compat ánh xạ warp = nửa wavefront.
Mọi cây giảm 32 lane chạy ~5 shuffle cho việc lẽ ra là 1 butterfly 64 lane. `reduce12_gfx906` đã đo
"16 shuffle exchanges instead of 60" cho đúng loại việc này, trên decode. Việc của prefill: liệt kê
kernel theo thời gian bằng `rocprof --stats` trên một prompt, đổi sang butterfly 64 lane ở đầu danh
sách — **chỉ bit-exact khi thứ tự giảm không đổi**, và phải chứng minh điều đó từng kernel (64 lane
gộp hai nửa 32 lane không cùng thứ tự kết hợp).

---

## 7. Những gì chưa kiểm chứng trong tài liệu này

- Bảng mục 5 là **phép tính** từ 55.4 GiB và 13.7 GB/s; **chưa** có lần quét `--prefill` nào trên
  MI50 32 GB, và 13.7 GB/s lấy từ Radeon Pro VII chứ không phải từ card đo chính.
- Cột "gfx906 chưa đo" ở các dòng C, D, J: đường FP16-in/out, `STRATA_PF_GEMM`/`STRATA_PF_PAD`,
  `STRATA_SELECT_SIMT` — cả ba đều **chưa** có số trên MI50.
- ≈50 MFLOP/token cho attention tính từ shape mặc định trong `geometry.hpp`, **chưa** đọc từ
  checkpoint thật.
- 40 token/expert ở N1 là số trung bình từ 10/512 × chunk; **phân phối thật** (độ lệch, số expert
  nhận 0 token) chưa đo.
- Không có số end-to-end nào của các hướng N1–N5: chúng là đề xuất, chưa phải kết quả.
