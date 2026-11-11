# Kế hoạch cải tạo MI50: dự án prefill trên gfx906

Đây là kế hoạch thực hiện, dẫn xuất từ phân tích ở
[GFX906_PREFILL_METHODS.md](GFX906_PREFILL_METHODS.md). Đọc tài liệu đó trước; ở đây chỉ có việc
phải làm, theo thứ tự, với tiêu chí pass/fail cho từng bước.

## 0. Dự án này là gì, và không là gì

**Là:** prefill trên Instinct MI50 (gfx906) đạt **1,000 tok/s**, bằng mọi thuật toán, không dùng CUDA.

Mục tiêu đó **không nằm trong trần FP32 scalar của card**, và đó là sự thật định hình cả kế hoạch:

| | |
|---|---|
| Cần | 1,000 tok/s × ~12 GFLOP/token (6B hoạt động) = **~12 TFLOPS** |
| Đỉnh `v_fma_f32` một MI50 | **~7.4 TFLOPS** (60 CU × 64 lane × 2 × xung nhân) |
| Trần suy từ đỉnh đó | **~617 tok/s** |
| Đo hiện tại | 330–350 tok/s, và GEMM FP16 cô lập đã **93–99% đỉnh** tại `T ≥ 512` |

Vì vậy tối ưu pha (N1, N2, N5, switch) chỉ đưa tới ~560–617 tok/s. **1,000 tok/s chỉ đến từ việc phá
một trong bốn giả định** — xem mục 9. Toàn bộ giai đoạn 0–7 vẫn phải làm, vì chúng là nền cho cả bốn
đường; nhưng mục 9 mới là phần quyết định dự án có đạt 1k hay không.

> Ghi chú ngày 21/10/2026: tôi không truy cập được web trong phiên viết tài liệu này
> (WebSearch/WebFetch bị chặn), nên mọi con số dưới đây lấy từ mã và từ các phép đo đã ghi trong repo,
> cộng với spec công bố của AMD mà **chưa đối chiếu lại nguồn ngoài**. Hai chỗ repo *khẳng định* mà
> chưa đo, và chúng là chỗ gánh toàn bộ mục tiêu 1k: (a) "FP16 packed chạy cổng nửa tốc độ"
> ([GFX906_SINGLE_MI50_IQ3_S.md:161](GFX906_SINGLE_MI50_IQ3_S.md#L161), [:363](GFX906_SINGLE_MI50_IQ3_S.md#L363)),
> trong khi bảng thông số MI50 công bố ghi 13.1 TFLOPS FP16 = **2×** FP32; (b) không một kernel nào
> trong repo dùng số học packed — `grep` toàn repo không có `__hfma2` hay `half2` FMA nào, mọi đường
> đều mở half ra rồi nhân `float`. Hai chỗ đó phải đo, không được tin.

**Không là:** đuổi theo các hướng mà cuộc đua CUDA đang chạy. Trên gfx906 những hướng đó không tồn
tại về mặt kiến trúc, không phải "chậm hơn":

| Người khác đang làm | Trên gfx906 |
|---|---|
| Tensor core / WMMA / MFMA GEMM | Không có khối đó. `dev_info()` ở [moe_fused_iq.cu:1258](../src/prefill/moe_fused_iq.cu#L1258) trả `DevInfo` rỗng ⇒ kernel `mma.sync` không bao giờ được gọi |
| INT8 / FP8 `V_DOT` dot-product | gfx906 không có `V_DOT`; `udot4` xuống chuỗi `v_macx_u8` ⇒ int8 **ngang nhịp** FP32. Đã đo: MMQ int8 chậm hơn FP16 **3.8–12.4× ở mọi `T`** |
| FP16 packed gấp đôi FP32 | **Chưa biết, và đây là câu hỏi mở lớn nhất của dự án.** Repo khẳng định
  `v_pk_fma_f16` chạy cùng nhịp FP32 nhưng **chưa từng đo**, và chưa kernel nào trong repo gọi packed
  FMA. Bảng thông số công bố của MI50 ghi 13.1 TFLOPS FP16 = 2× FP32. Đo bằng
  `gfx906_fma_rate_probe` (mục 9A) trước khi tin một phía nào |
| hipBLASLt tuned algo | roc community image không dùng hipBLASLt ([AMD_HIP.md:657](AMD_HIP.md#L657)); đường dense là rocBLAS |
| cp.async / TMA staging LDS | Không có; mỗi load LDS là VMEM → register → LDS. Đây là ràng buộc thiết kế kernel, không phải tuỳ chọn |

Kết luận phương pháp, có điều kiện: với **đường FP32 scalar** hiện tại, FLOP đã hết room (GEMM cô lập
93–99% đỉnh), nên phần dọn pha (`dequant`, launch ragged, chờ copy, cây giảm 32 lane) là đúng và phải
làm. Nhưng **con đường 1k không đi qua đó** — nó đi qua mục 9: hoặc tìm ra một đường toán nhanh hơn
một FMA scalar một nhịp, hoặc cắt FLOP mỗi token, hoặc tính ít token hơn, hoặc thêm card. Chạy probe
trước, rồi hãy quyết định đầu tư bao nhiêu tuần vào kernel.

## 1. Nguyên tắc làm việc (không thương lượng)

Theo [AGENTS.md](../AGENTS.md) và [AMD_HIP.md](AMD_HIP.md):

1. **Một thay đổi một PR.** Nói rõ nó đổi gì và để nguyên gì.
2. **Tính năng mới là opt-in**; đường mặc định giữ **byte-identical** với bản release cuối, cho tới
   khi có số đo.
3. **Parity ba lớp trước khi nói "nhanh hơn"**: (a) ctest parity trên dữ liệu tổng hợp, (b)
   `native_expert_bench` / `mmq_expert_bench` trên dòng GGUF thật, (c) văn bản greedy của một request
   cố định trên server đang chạy. Nếu thứ tự cộng đổi (butterfly 64 lane, split theo K), **phải**
   chứng minh bit-exact từng kernel hoặc chấp nhận khác bit và nói rõ.
4. **Đổi default chỉ ở nơi đã đo là nhanh hơn**, kèm số và kèm tên máy.
5. **Build mọi backend mà file chạm tới** (CUDA, HIP, SYCL) trước khi xin review.
6. Mỗi mục dưới đây tự nó là một PR; không gộp hai mục vào một PR, kể cả khi cùng một file.

Máy đo chuẩn cho toàn dự án: **một MI50 32 GB**, 60 CU, ROCm 6.3.3 local + image
`mixa3607/rocm-gfx906:7.14-complete`, `build-906`, pack IQ3_S `--native`, config khởi điểm ở
[GFX906_SINGLE_MI50_IQ3_S.md](GFX906_SINGLE_MI50_IQ3_S.md) mục 3. Số của card khác (Pro VII 16 GB,
2×MI50 16 GB) chỉ được dùng để đối chiếu và phải ghi rõ là của máy nào.

## 2. Giai đoạn 0 — dựng thước đo (1–2 ngày, không đổi code)

Không bắt đầu viết kernel nào khi chưa có bảng này. Đây là PR duy nhất **không** có thay đổi hành vi.

| Việc | Lệnh | Sản phẩm |
|---|---|---|
| 0.1 Phân bổ thời gian prompt theo pha | `STRATA_PREFILL_TIMING=1` trên prompt 4K và 32K, `--prefill 2048` | Bảng 20 pha (`PfPhase`, [prefill.cpp:2287](../src/prefill/prefill.cpp#L2287)) cho mỗi prompt. Đây là con số gốc của cả dự án |
| 0.2 Đếm kernel và thời gian mỗi kernel | `rocprof --stats` trên cùng một run | Xếp hạng kernel; số launch mỗi layer |
| 0.3 **Đo PCIe trên chính máy đo** | copy một blob pinned 4 GiB host→device | Thay con số 13.7 GB/s đang mượn từ Radeon Pro VII. Mọi tính toán chunk phụ thuộc nó |
| 0.4 Quét chunk | `--prefill` 1024 / 2048 / 3072 / 4096 / 6144, mỗi cỡ ba lần | Đường tok/s theo chunk, kèm dung lượng đỉnh VRAM. Kiểm chứng (hoặc bác bỏ) bảng mục 5 của tài liệu phân tích |
| 0.5 Quét ring + hit rate | `STRATA_PREFILL_RING` quanh giá trị hiện tại trên IQ3_S | Bảng slot vs tok/s; hit rate expert cache trên IQ3_S |
| 0.6 Tách ba thay đổi chưa đo của commit `10230425` | ba bản build trong `build-906/`: baseline, chỉ-`udot4`, chỉ-cờ-inline | Số quy lỗi cho `udot4`, `-mllvm -amdgpu-early-inline-all`, và mode 8 |

**Gate của giai đoạn 0:** bảng pha 0.1 + đường chunk 0.4 + PCIe 0.3. Nếu hit rate IQ3_S < 0.95 ở 0.5,
**dự án đổi thứ tự**: chuyên mục 4 (cache/byte) lên trước, vì kernel không phải nút thắt.

Điểm đối chiếu đã có, để biết mô hình byte đúng tới đâu: trên **2×MI50 16 GB** với **toàn bộ 12,288
expert nằm trong VRAM**, prompt 17,043 token ở `--prefill 4096` chạy **392 tok/s** (IQ1_M,
[AMD_HIP.md](AMD_HIP.md)). Không có streaming ở run đó, nên nó là control cho phía FLOP, không phải
phía byte.

## 3. Giai đoạn 1 — tiền rẻ: switch và chính sách (3–7 ngày)

Mỗi dòng một PR, tất cả đã có sẵn trong mã, chỉ cần đo và (nếu thắng) đổi default.

| PR | Việc | Tiêu chí |
|---|---|---|
| 1a | `--prefill`: đặt chunk theo công thức `55.4 GiB / C < 0.85 × PCIe_đo_được` (mục 5 tài liệu phân tích), thay vì theo bộ nhớ còn lại | Chỉ merge nếu 0.4 cho thấy tok/s tăng thật trên 32 GB. Ghi rõ 16 GB không đổi |
| 1b | `STRATA_SELECT_SIMT=1` (bộ chấm điểm QSA FP32 tiled) | gfx103x +7.4% ở 128K; gfx906 **chưa đo**. Đây là đường tự nhiên của card không có tensor core |
| 1c | `STRATA_GFX906_ATTN_REDUCE12=1` + `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` | `build-906/gfx906_qsa_query_swizzle_probe` + `bench/test_gfx906_qsa_query_swizzle_guard.py` trên 4K/32K/128K. Guard ở [qsa_decode_attn.cu:824](../src/kernels/cuda/qsa_decode_attn.cu#L824) luôn đúng với MI50 |
| 1d | `STRATA_HIP_PROMPT_F16=1` | Đo riêng trên gfx906. **Kỳ vọng thấp**: FP16 cùng nhịp FP32, chỉ tiết kiệm băng thông ghi `Y`. Không bit-exact ⇒ ở lại opt-in |
| 1e | `STRATA_PF_GEMM=1` + `STRATA_PF_PAD=1` | Đo riêng trên gfx906; con số 19→29 TFLOPS trong ghi chú code là **card khác**, không được dùng làm lý do |
| 1f | `STRATA_ROUTE_PER_TOKEN=1`, `STRATA_GR_FAST` giữ bật, `STRATA_MMVQ_WAVE` A/B | Mỗi cái một run riêng. `STRATA_GR_SPLIT` không bit-exact ⇒ chỉ bật nếu chấp nhận khác bit |

Ghi chú cho 1d/1e: hai PR này có thể **kết luận bằng một commit tắt** (giống cách
`STRATA_GFX906_MMQ_MAX_T` 512→0 ở [gemm.cu:1273](../src/prefill/gemm.cu#L1273)). Một PR "đo và tắt"
là kết quả hợp lệ, và là kết quả tốt hơn một con số không giải thích được.

## 4. Giai đoạn 2 — N2: giết pha `dequant` (2–4 tuần, giá trị cao nhất)

**Giả thuyết:** mỗi GEMM expert hiện đọc weight nén (~0.43 B/weight IQ3_S), **viết** 13.1 MB FP16
cho một expert gate/up 1280×2560, rồi GEMM **đọc lại** 13.1 MB đó. Expert không dùng lại giữa chunk,
nên FP16 scratch là một vòng qua HBM thuần túy vô ích. Thu được: gần hết pha `dequant` + phần `gemm`
vì weight đọc một lần ở dạng nén.

**Việc:**
1. Đo cô lập chi phí `dequant_f16` cho hai shape (`iq3_s gate/up` N1280/K2560, `iq4_nl down`
   N2560/K640) ở `T` = 128/256/512/1024. Mở rộng `gfx906_mmq_bench` (đã có trong
   [gfx906_mmq_bench.cpp](../src/prefill/gfx906_mmq_bench.cpp)) thành `gfx906_dequant_probe`: in riêng
   thời gian dequant và thời gian GEMM, thay vì chỉ tổng.
2. Kernel fused: weight block GGUF → **LDS tile FP16** (32×32 = 2 KiB, ×4 bộ đệm; LDS 64 KiB/CU là
   đủ nhưng không còn dư), FMA từ LDS, tích luỹ nhiều accumulator (kiểu đã làm ở commit `d33f45e3`).
   Không `cp.async`: phải tự mồi load sớm hơn và để compiler thấy tính độc lập.
3. Cùng kernel đó cho dense projections nếu `N*K` vào được scratch; nếu không, giữ đường slice.

**Gate:** parity bitwise với đường hiện tại trên cả hai shape (cùng thứ tự cộng), `mmq_expert_bench`
trên GGUF thật, rồi greedy text. Chỉ báo cáo con số khi cả ba lớp xanh. Merge opt-in
(`STRATA_GFX906_FUSED_DQ=1`), chỉ đổi default khi có số end-to-end trên MI50 32 GB.

## 5. Giai đoạn 3 — N1: GEMM expert theo bucket có đệm + bảng tile (3–6 tuần)

**Giả thuyết:** ở chunk 2,048, mỗi expert nhận trung bình `10/512 × 2048 ≈ 40` token — rất xa điểm
93–99% đỉnh đo ở `T ≥ 512` — và 512 launch ragged mỗi layer trả giá latency LDS mà gfx906 không có
cách giấu. Không có bằng chứng thực nghiệm cho **phân phối thật** số token mỗi expert; đó là bước 1.

**Việc:**
1. **Đo phân phối trước khi thuật toán hoá**: histogram số token mỗi expert cho mỗi layer, ở chunk
   1024/2048/4096, trên một prompt 4K và một prompt 32K. Không có số này, mọi con số lợi nhuận đều
   là suy đoán.
2. Counting-sort token theo expert → **đệm lên bucket luỹ thừa hai** (16/32/64/128) → **một launch
   mỗi bucket**, với bảng trỏ tile → `(expert, row-tile)`. Số launch mỗi layer xuống ~4–6.
3. So với đường grouped hiện tại (mode 8) trên cùng GGUF. Báo cáo cả phần đệm thừa trung bình, vì đó
   là chi phí thật của thuật này.

**Gate:** parity bitwise (thứ tự cộng phải giữ nguyên — nếu bucket đổi thứ tự tile thì phải chứng
minh, hoặc opt-in kèm KL). Lợi nhuận tối thiểu để merge: **+8% end-to-end prefill** trên 32 GB; dưới
ngưỡng đó, đóng hướng và ghi lại lý do.

## 6. Giai đoạn 4 — N5: dọn reduction và kernel nhỏ (2–3 tuần)

Lớp HIP-compat ánh xạ warp CUDA = nửa wavefront 64 lane, nên mỗi cây giảm 32 lane tốn ~5 shuffle
cho việc lẽ ra là 1 butterfly 64 lane, và `__syncwarp` = fence + `wave_barrier` + fence trên cả 64
lane. `reduce12_gfx906` đã đo "16 shuffle exchanges instead of 60" trên decode.

**Việc:** lấy danh sách kernel từ 0.2 (prompt, không phải decode), đổi cây giảm sang butterfly 64 lane
theo thứ tự thời gian; sau đó gộp các kernel nhỏ còn lại (`hc read`, `combine`, norm, `embed+steps`)
theo cùng cách mode 8 đã làm cho gate/up + SwiGLU + q8_1.

**Gate:** từng kernel một, parity + số đo riêng. Một kernel một PR. Không gộp "dọn dẹp".

## 7. Giai đoạn 5 — N3: cache FP16 cho expert thường trú (chỉ nếu giai đoạn 0 nói byte là nút thắt)

Giữ bản FP16 đã giải nén của expert resident trong VRAM: đổi **4.6× byte VRAM** lấy **0 lần dequant**.
Chỉ hợp lý cho expert resident trên card 32 GB; trên 16 GB gần như chắc chắn sai vì nó ăn slot cache.
Quyết định làm hay không **nhờ** số hit rate ở 0.5, không đoán.

## 8. Những hướng đã đóng, và vì sao đóng

| Hướng | Tình trạng |
|---|---|
| INT8 MMQ trên gfx906 (`moe_mmq_gfx906`, `STRATA_DENSE_MMQ`) | **Đóng** bằng số đo 19/10: chậm hơn 3.8–12.4× ở mọi `T`. Lý do kiến trúc: không `V_DOT`. Chỉ mở lại nếu kernel có LDS tile 2D **và** một lý do byte, không phải lý do FLOP |
| Mục tiêu "≥1,000 tok/s prefill" | **Mở lại, và là mục tiêu của dự án.** Bất khả thi *trong phạm vi một card với số học FP32 scalar và 12 GFLOP/token*: cần ~12 TFLOPS = 1.6× đỉnh. Đạt được nếu phá một giả định — xem mục 9 |
| "52.2 TOPS INT8 / 26.8 TFLOPS FP16" cho MI50 | **Sai kiến trúc** (bảng phủ nhận trong tài liệu IQ3_S mục 6b). Không dùng làm đầu vào cho bất kỳ tính toán nào |
| bf16 HCD, `rdna2_sgemm`, WMMA GEMM, GDN chunked | Không tồn tại trên gfx906 (mục 2 tài liệu phân tích) |
| Tối ưu QSA attention cho prompt | **Xếp sau cùng**: sparse width tối đa 2,051 cell **không đổi** theo độ dài prompt ⇒ ≈50 MFLOP/token, **<1%** FLOP mỗi token. Đừng bắt đầu ở đây |
| CUDA fallback cho bất kỳ phần nào | Loại theo yêu cầu của dự án. gfx906 build là một backend riêng (`STRATA_HIP_GFX906`), không kèm `STRATA_ENABLE_HIP` |

## 9. Con đường tới 1,000 tok/s: bốn đòn bẩy, theo thứ tự đáng làm

`1,000 tok/s × 12 GFLOP/token = 12 TFLOPS`. Một MI50 cho 7.4. Không có cách nào lấy 12 từ 7.4 bằng
cách dọn thời gian chết. Phải phá một trong bốn giả định, và phải biết giả định nào phá được **trước**
khi đầu tư tháng nào.

### A. Phá giả định "một FMA một nhịp" — chưa đo, rẻ nhất, gánh cả dự án

Toàn bộ kết luận "617 tok/s là trần" đứng trên một câu chưa từng được đo: *FP16 packed chạy cổng nửa
tốc độ*. Repo chưa từng gọi `__hfma2` ở đâu cả — `grep` toàn repo không có packed FMA nào. Spec công
bố của MI50 lại ghi **13.1 TFLOPS FP16 = 2× FP32**. Nếu spec đúng và khẳng định trong repo sai, thì
trần là **~1,230 tok/s** và 1k nằm trong tầm tay của một kernel, không phải của một giấc mơ.

Việc, theo thứ tự:

1. `ninja -C build-906 gfx906_fma_rate_probe && ./build-906/gfx906_fma_rate_probe`
   ([gfx906_fma_rate_probe.cpp](../src/prefill/gfx906_fma_rate_probe.cpp), mới). Eight accumulator
   độc lập, không chạm HBM, in tỉ lệ `v_pk_fma_f16 : v_fma_f32` và `udot4 : v_fma_f32` ngay trên card.
   Đây là con số quyết định, đo trong vài phút.
2. Nếu `hfma2` > 1.5×: **mọi GEMM prompt phải viết lại quanh packed half2**, và `dequant` phải xuất
   `half2` (chẵn, kề nhau) chứ không phải `float`. N2 (mục 4) đổi thành *packed* dequant-into-LDS.
   Trần mới ~1,230 tok/s; M3 dưới đây thay bằng M-A.
3. Nếu `hfma2` = 1.0×: đóng đường A, ghi lại số đo để lần sau không phải đoán nữa, và 1k trên **một**
   card chỉ còn đường B hoặc C.

Cùng câu hỏi cho phần cứng anh em: **gfx908 (MI60 / S7150) có Matrix Core (MCE, MFMA)**, gfx906 thì
không. Đó vẫn là card AMD ROCm, vẫn không liên quan CUDA, 16 GB HBM2, và là con đường duy nhất để có
accelerator thật mà không rời hệ AMD. Nó là quyết định *chọn card*, không phải thuật toán — nhưng nếu
1k là bắt buộc, nó phải được nêu thẳng.

### B. Phá giả định "12 GFLOP mỗi token"

FLOP mỗi token là `2 × số tham số hoạt động`, không đổi được bằng quantization (quant đổi byte, không
đổi FLOP — đã chứng minh ở gfx906). Đổi được bằng thuật toán, mỗi cái đánh đổi chất lượng và phải đo
chất lượng chứ không chỉ đo tok/s:

| Cách | Cắt FLOP | Giá phải trả | PR |
|---|---|---|---|
| Giảm `k` chuyên gia ở pha prompt (top-k nhỏ hơn cho token prompt so với token decode) | ∝ k | Chất lượng trên prompt dài; đo bằng KL và greedy | 2 tuần |
| Bỏ qua một số lớp MoE với token prompt ít thông tin (layer skip / early-exit có ngưỡng) | tới 30–40% | Rủi ro nhất; cần chuẩn đo chất lượng cố định | 3–4 tuần |
| Dense projections thấp bit cho phần không phải expert | một phần nhỏ | Không đổi FLOP trên gfx906 → **vô nghĩa ở đây** | — |

Đây là đường duy nhất có thể cắt FLOP **mà không cần thêm phần cứng**. Nó cũng là đường mà cuộc đua
CUDA không chạy bằng, vì họ ẩn chi phí bằng tensor core.

### C. Phá giả định "mọi token của prompt phải được tính"

Prompt/prefix cache: nếu phần đầu prompt trùng một request trước, không tính lại nó. Không tăng tốc
phép tính nào, nhưng tăng **throughput hiệu dụng** — và với workload agent (prompt lặp lại phần lớn
system + tool schema) tỷ lệ trùng lặp là con số thật, chưa đo trên máy này.

Việc: đo tỷ lệ prefix trùng lặp trên workload thật trước, rồi quyết định có đáng không. Đây là cách
duy nhất để "1,000 tok/s" xuất hiện trên màn hình **trên một card gfx906, tuần này**. Nói rõ với người
dùng đó là throughput hiệu dụng, không phải tốc độ tính.

### D. Phá giả định "một card"

`docs/AMD_HIP.md` đã có số: **2×MI50 16 GB, toàn bộ 12,288 expert trong VRAM, 392 tok/s** với IQ1_M —
tức là đã có đường multi-GPU chạy được. Bốn MI50 = **29.6 TFLOPS FP32**; 1k cần 12 TFLOPS = **41%**
đỉnh bốn card, một hệ số sử dụng mà chính phép đo GEMM cô lập trên một card đã đạt (93–99%). Đây là
đường chắc chắn nhất về số học, và nó chỉ bị ràng buộc bởi topologia + chi phí allreduce.

Việc: đo hiệu quả scale 1→2 card trên cùng IQ3_S 32 GB (đường `MULTI_GPU.md`), rồi quyết định xem 3–4
card có đáng so với đường A. `docs/MULTI_GPU.md` chưa có số cho gfx906.

### Thứ tự quyết định

```
A-probe (vài phút, đã có code) ──> hfma2 2×? ── CÓ ──> kernel packed-FP16 (N2 đổi shape) ──> M-A
                                        │
                                        NO ──> B (cắt FLOP/token) + C (prefix cache) + D (thêm card)
```

Không làm B hay D trước khi chạy probe. Probe rẻ hơn mọi con đường khác và nó quyết định trần của cả
dự án.

## 9b. Mốc thành công

| Mốc | Tiêu chí | Ghi chú |
|---|---|---|
| M0 | Bảng pha + đường chunk + PCIe trên máy đo, **và `gfx906_fma_rate_probe` in ra tỉ lệ `hfma2`/`udot4`** | Không đổi hành vi. Probe là con số đầu tiên của dự án |
| M1 | prefill ≥ **420 tok/s** trên IQ3_S 32 GB | = ~5.7 TFLOPS, chỉ bằng switch/chính sách |
| M2 | prefill ≥ **500 tok/s** | = ~6.7 TFLOPS, sau N2 |
| M3 | prefill ≥ **560 tok/s** = **90% đỉnh FP32 scalar** | è trần của đường "một card, FP32 scalar, 12 GFLOP/token". Sau N1 + N5 |
| M-A | Nếu probe nói packed FP16 nhanh hơn: prefill ≥ **1,000 tok/s** trên một MI50 | = ~12 TFLOPS ≈ 92% đỉnh FP16 packed. Cần kernel packed mới + dequant packed |
| M-B | Nếu không: 1k chỉ khả thi với B (cắt FLOP/token) hoặc D (2–4 card) hoặc C (throughput hiệu dụng) | Mỗi cái phải kèm báo cáo chất lượng hoặc chi phí phần cứng |
| M4 | Toàn bộ parity ba lớp xanh trên MI50 32 GB; đường mặc định của card khác không đổi một byte | Điều kiện để các PR được review |

M3 không còn là đích cuối. Đích cuối là **M-A**, và nó phụ thuộc vào một phép đo chưa chạy.

## 10. Rủi ro chính

1. **Mô hình byte (mục 5 tài liệu phân tích) là phép tính, chưa phải phép đo.** Giai đoạn 0.3/0.4 có
   thể bác nó. Nếu bị bác, thứ tự giai đoạn 1–5 đổi theo — đó là lý do giai đoạn 0 đứng trước.
2. **Bit-exact là ràng buộc cứng.** Butterfly 64 lane và split theo K đổi thứ tự cộng. Mỗi chỗ phải
   chứng minh hoặc opt-in; không có chỗ "chắc là được".
3. **Dung lượng.** Chunk ≥ 3,072 và N3 đều ăn VRAM trên đúng card đang thiếu VRAM. Mọi PR tăng dung
   lượng phải báo cáo dung lượng đỉnh cùng với tok/s.
4. **Số IQ3_S cộng đồng là IQ2_XS**; số PLE 26.8 GiB là IQ4_NL; mỗi layer dùng format khác nhau
   (iq3_xxs, iq2_s, iq4_nl cho down). Một con số "IQ3_S" duy nhất không tồn tại ở mức kernel.
5. **Không có card gfx906 trong tay maintainer** ([AMD_HIP.md:649](AMD_HIP.md#L649)). Mọi PR phải kèm
   báo cáo theo [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md), so với một run cùng ngày của chính
   bản build đang test, và nói rõ cái gì **không** được test.
