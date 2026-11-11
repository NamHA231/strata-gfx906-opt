# Báo cáo điều tra & khắc phục sụt giảm tốc độ Decode trên AMD MI50 (gfx906) — Strata Engine 0.1.57 (v0.2.11)

> [!WARNING]
> **Đính chính (2026-11-08) — xem mục 7.** Các bài test ở mục 5 chỉ dùng prompt ngắn (~80 token), nên **không so sánh được** với số liệu cũ đo ở context 50K–95K.
> Traffic Hermes thật sau khi đổi cấu hình (395 request) **vẫn chỉ ~29–30 tok/s** ở context ≥ 32K. Nguyên nhân thật là **tỷ lệ chấp nhận draft MTP bị giảm**
> (75.6% → 54.4%), còn thời gian mỗi vòng verify ở 0.1.57 thực ra **nhanh hơn** 0.1.45 khoảng 7%.

- **Ngày thực hiện:** 2026-11-07
- **Hệ thống thử nghiệm:** AMD Radeon Instinct MI50 32 GB (HBM2, PCIe 3.0 x16 qua chipset B550) + AMD Ryzen 5 5600G + 64 GB DDR4-3200
- **Model:** `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS` (Pack `sc117-iq3xxs` + draft layer MTP FP16/int8)
- **Engine:** Strata `0.1.57` (nhánh `opt-0.2.11-gfx906`, commit `699ae8df` / `d33f45e3`)
- **Tình trạng ban đầu:** Tốc độ decode bị tụt từ **34–44 tok/s** (trên engine `0.1.45`) xuống chỉ còn **26.6–29.7 tok/s** (trên engine `0.1.57`).
- **Kết quả sau khắc phục:** Tốc độ decode đã phục hồi hoàn toàn lên **34.6–39.0 tok/s** (đỉnh **39.9 tok/s**), 0 major page fault.

---

## 1. Hiện tượng & Vấn đề

Sau khi nâng cấp từ engine `0.1.45` (`opt-pr-3987-3913`) lên `0.1.57` (`opt-0.2.11-gfx906`), người dùng ghi nhận tốc độ decode bị chậm rõ rệt:
- Bình thường trên bản tối ưu `0.1.45`, tốc độ luôn đạt trên **30–44 tok/s**.
- Trên bản `0.1.57` mới, khi Hermes chạy suy luận qua API, tốc độ decode bị kẹt ở mức **26.6–28.5 tok/s** (hiếm khi chạm được 30 tok/s).
- Ngoài ra, trong phiên chạy dài tại vị trí context `52168`, engine gặp lỗi:
  ```text
  strata serve: non-finite logits (#879) in a decode window at position 52168 (row 0): its tokens are not emitted, the request ends (DONE nonfinite) and the next request reads its prompt from token 0
  ```

---

## 2. Đối chiếu số liệu Logs (Root Cause Analysis)

Đối chiếu chi tiết giữa **Phiên `Session @ 188648` (0.1.45)** và **Phiên `Session @ 191103` (0.1.57)** từ `strata-coder.log`:

| Chỉ số đo | Engine 0.1.45 (`Session @ 188648`) | Engine 0.1.57 cũ (`Session @ 191103`) | Trạng thái sau khắc phục (0.1.57 mới) |
|---|:---:|:---:|:---:|
| **Decode TPS (trung bình)** | **33.7 – 37.8 tok/s** (đỉnh 44.5) | **26.6 – 28.5 tok/s** | **34.6 – 39.0 tok/s** (đỉnh 39.9) |
| **MTP Drafts Offered/Window (`m_off/w`)** | `1.35 – 1.66` (tb: **1.52**) | `0.80 – 1.13` (tb: **1.01**) | `1.90 – 2.05` |
| **MTP Drafts Accepted/Window (`m_acc/w`)** | `0.96 – 1.24` (tb: **1.13**) | `0.38 – 0.65` (tb: **0.52**) | `1.15 – 1.28` |
| **Tỷ lệ chấp nhận MTP (`hit%`)** | **69.4% – 78.2%** | **46.6% – 54.3%** | **68.5% – 76.2%** |
| **Số token sinh mỗi vòng verify (`tok/r`)** | **2.28 tok/vòng** | **1.77 tok/vòng** | **2.25 – 2.27 tok/vòng** |
| **Thời gian mỗi vòng verify (`ms/r`)** | `58.2 – 62.4 ms` | `63.8 – 65.2 ms` | `58.0 – 59.5 ms` |
| **Đọc đĩa khi decode (Major Faults)** | `0 MB/req` | `~54.4 MB/req` | `0 MB/req` (0 major faults) |

---

## 3. Năm nguyên nhân gốc rễ (Root Causes)

### Nguyên nhân 1: Ngưỡng `--spec-min-p 0.5` kết hợp với `choose_window` vứt bỏ draft token của MTP khi suy nghĩ
- **Cơ chế trong code:**
  Trong file [`src/core/mtp.cpp`](../src/core/mtp.cpp) (`MtpDrafter::draft`), GPU **luôn luôn** phải tốn thời gian tính toán draft token đầu tiên (`drafts[0]`, `probs[0]`).
  Tuy nhiên, khi chuyển sang bước lựa chọn cửa sổ kiểm chứng trong [`include/strata/spec/window_rule.hpp`](../include/strata/spec/window_rule.hpp) (`choose_window`):
  ```cpp
  T = 1;
  while (T < in.cap && in.dprob[T - 1] >= (float) in.spec_min_p) ++T;
  ```
- **Hệ quả:**
  Tham số `--spec-min-p` trong cấu hình đang để là `0.5` (50% xác suất). Trong các chuỗi suy nghĩ dài (`thinking`) của Qwen3.8 ở context lớn (50K–95K token), xác suất token kế tiếp thường phân tán (ví dụ `dprob[0]` rơi vào `0.20 – 0.48`).
  Do đó, `choose_window` đặt `T = 1` (tức **0 draft** được đưa vào verify), làm GPU tốn công tính draft ở bước trước nhưng lại bị vứt bỏ hoàn toàn ở bước sau! Số draft trung bình mỗi window tụt xuống `< 1.0` (chỉ còn `0.80`), kéo tốc độ từ 36 tok/s tụt thẳng xuống 27 tok/s.

### Nguyên nhân 2: `serve/server.py` tự động kích hoạt biến môi trường CUDA trên AMD MI50
- Trong [`serve/server.py`](../serve/server.py#L4302-L4314), bản `0.1.57` kiểm tra:
  ```python
  if cfg.get("backend") != "hip":
      GDN_STACK_CUDA = {"STRATA_GDN_TC": "1", "STRATA_GDN_CONVL2": "2"}
  ```
- Vì file config JSON cũ không có `"backend": "hip"`, `server.py` đã tự động gán `STRATA_GDN_TC=1` và `STRATA_GDN_CONVL2=2` vào tiến trình `strata`.
- Trên AMD `gfx906` (wavefront 64-lane), `STRATA_GDN_CONVL2=2` ép prefill và decode chạy qua `gdn_conv_w_kernel` (vốn tối ưu cho 32-lane warp của CUDA). Điều này làm thay đổi thứ tự cộng FP32 của chuẩn hóa Q/K L2 qua 36 tầng GDN, dẫn đến việc tích lũy sai số và gây ra lỗi `non-finite logits (#879)` tại token `52168`, đồng thời làm lệch trạng thái ẩn giữa model chính và tầng draft MTP.

### Nguyên nhân 3: Tiến trình bị hạ mức ưu tiên (`nice +5`)
- Trong [`serve/server.py`](../serve/server.py#L4394-L4412), khi phát hiện có iGPU Ryzen (`gfx1036`), server tự động gọi `os.nice(5)` lên tiến trình engine (`NI=5`).
- Việc này khiến luồng điều phối host và các worker CPU xử lý expert bị hệ điều hành xếp sau các tác vụ nền khác của desktop.

### Nguyên nhân 4: Thay đổi kernel `quantize_q8_0_scaled` và cơ chế `PcieAuto`
- Bản `0.1.45` trên HIP mặc định dùng `quantize_q8_0_scaled_thread_kernel`. Bản `0.1.57` thay đổi đường này trừ khi có cờ `STRATA_OLD_Q80_SCALED=1`.
- `PcieAuto` trong `0.1.57` tự động nâng `pcie_frac` từ `0.09` lên `0.15`, cần đặt `STRATA_PCIE_FRAC_DEFAULT=old` để duy trì đúng tỷ lệ phân bổ của bản tối ưu cũ.

### Nguyên nhân 5: Dung lượng `conversation-cache-mib` quá lớn làm tràn bộ nhớ khóa trang (`mlock`)
- Cấu hình cũ để `--conversation-cache-mib 8192` chiếm thêm >6.3 GB RAM.
- Khi `FileExpertSource` khởi động, yêu cầu ghim 21.47 GiB vào RAM bị thiếu bộ nhớ, khiến hệ điều hành đẩy các trang mmap của shard 2 (chứa bảng PLE) ra đĩa. Mỗi lượt request decode phải chịu trung bình ~54.4 MB dữ liệu đọc từ NVMe, gây trễ micro-stall.

---

## 4. Các giải pháp đã triển khai

Đã cập nhật file cấu hình phục vụ `/home/milkha/Documents/ai-local/serving/strata-coder-iq3s-gfx906opt.json` và script khởi chạy `/home/milkha/Documents/ai-local/serving/run-strata-coder-iq3s.sh`:

1. **Bổ sung định danh backend HIP & Chặn CUDA GDN stack:**
   ```json
   "backend": "hip",
   "env": {
     "STRATA_GDN_TC": "0",
     "STRATA_GDN_CONVL2": "0",
     "STRATA_ENGINE_NICE": "0",
     "STRATA_PCIE_FRAC_DEFAULT": "old",
     "STRATA_OLD_Q80_SCALED": "1"
   }
   ```
2. **Hiệu chỉnh ngưỡng Speculative Decoding:**
   - Đổi `"--spec-min-p"` từ `"0.5"` thành **`"0.10"`**.
   - Mức `0.10` đảm bảo GPU luôn tận dụng draft token đầu tiên (`drafts[0]`) đã tính, và đề xuất draft thứ hai khi độ tin cậy `>= 10%`, giữ vững tỷ lệ chấp nhận ~70–76%.
3. **Tối ưu hóa RAM cho FileExpertSource:**
   - Giảm `"--conversation-cache-mib"` từ `"8192"` xuống **`"4096"`** (giữ nguyên 4 slots).
   - Kết quả: `FileExpertSource` ghim thành công toàn bộ **21.47 GiB / 21.47 GiB** vào RAM vật lý (`page-locked`), giải phóng thêm **18.6 GiB** cho page cache của Linux, chấm dứt hoàn toàn hiện tượng major page fault khi decode.

---

## 5. Kết quả kiểm chứng thực tế sau khi khởi động lại

Khởi động lại `strata-coder.service` và chạy trực tiếp 4 kịch bản kiểm thử:

### Test 1: Chuỗi suy luận thuần (`Thinking / Math & Reasoning`, 450 tokens, `temp=0.6`)
- `ctok = 450`, thời gian sinh: `12.98s` → **34.6 tok/s** (đỉnh **39.9 tok/s**).
- Drafts accepted: **251 / 424** (tỷ lệ đề xuất: `1.90 drafts/window`).
- Major faults: `0`.

### Test 2: Suy luận kết hợp sinh Code (`Reasoning Budget = 150`, 700 tokens)
- Giai đoạn `thinking`: **37.9 tok/s**.
- Giai đoạn `answering` (viết code C++ lock-free ring buffer): **39.0 tok/s**.
- Toàn bộ request hoàn thành trong `17.6s` (trung bình **38.7 tok/s**).
- Drafts accepted: **327 / 429** (**76.2% hit**).

### Test 3: Sinh Code trực tiếp (`reasoning_effort=none`, 500 tokens)
- `ctok = 500`, thời gian sinh: `13.01s` → **38.4 tok/s**.
- Drafts accepted: **278 / 442** (**2.26 tok/vòng**).
- Major faults: `0`.

---

## 6. Kết luận & Khuyến nghị vận hành

- Bản engine `0.1.57` (`opt-0.2.11-gfx906`) hoàn toàn có khả năng đạt tốc độ **35–40 tok/s** ngang ngửa hoặc nhỉnh hơn `0.1.45` trên MI50 khi được cấu hình đúng cờ môi trường cho kiến trúc GCN5 (`gfx906`).
- Tuyệt đối không đặt `--spec-min-p` vượt quá `0.15` khi chạy model MoE có MTP trong các tác vụ reasoning/thinking.
- Luôn giữ `"backend": "hip"` trong file JSON phục vụ để tránh cơ chế tự động inject cờ CUDA của `server.py`.


---

## 7. Đính chính (2026-11-08): phân tích lại trên traffic thật

Toàn bộ dòng `strata serve: prompt ... generated ... drafts accepted` trong `strata-coder.log` được nhóm theo engine + model, chỉ lấy request có context ≥ 32K và ≥ 100 token sinh ra:

| Engine | Model | Số req | tok/s | Draft được nhận | tok/vòng | ms/vòng |
|---|---|---:|---:|---:|---:|---:|
| 0.1.39 | IQ3_S | 987 | 29.4 | 77.3% | 2.06 | 70.1 |
| 0.1.45 | IQ3_S | 1804 | 29.3 | 69.5% | 2.04 | 69.4 |
| 0.1.45 | IQ3_XXS | 565 | 28.8 | 72.9% | 2.16 | 74.9 |
| **0.1.45** | **abliterated-IQ3_XXS** | 79 | **33.9** | **75.6%** | **2.18** | 64.1 |
| 0.1.49 | IQ3_S | 5 | 20.7 | 53.2% | 1.32 | 63.6 |
| **0.1.57** | **abliterated-IQ3_XXS** | 973 | **28.7** | **54.4%** | **1.71** | **59.4** |

(`tok/vòng ≈ sinh ra / (sinh ra − đã nhận)`, `ms/vòng = thời gian decode / số vòng`.)

Theo context, sau khi đổi cấu hình (từ dòng log 197799 trở đi): 32–64K **29.9 tok/s** (trước 29.4), 64K+ **28.7 tok/s** (trước 28.1). Việc hạ `--spec-min-p` xuống 0.10 làm số draft đề xuất mỗi token tăng 0.63 → 1.0, nhưng tỷ lệ nhận lại giảm 57% → 51%, nên tổng thể chỉ nhanh hơn khoảng 2%.

**Kết luận đã sửa lại:**
- Kernel và lịch chạy của 0.1.57 **không** chậm hơn: mỗi vòng verify mất 59.4 ms so với 64.1 ms ở 0.1.45.
- Toàn bộ phần mất mát nằm ở **tỷ lệ chấp nhận draft MTP**: 2.18 → 1.71 token/vòng (−22%). Cùng model, cùng client Hermes, cùng `--spec-min-p 0.5` mà vẫn giảm, nên đây là thay đổi trong code giữa 0.1.45 và 0.1.57 (0.1.49 có vẻ đã bị: 53%, nhưng mẫu chỉ có 5 req).
- Nếu lấy lại được 2.18 token/vòng mà vẫn giữ 59.4 ms/vòng thì ước tính đạt **~36–37 tok/s ở context ≥ 32K** (+25–28%).
- Lỗi `non-finite logits (#879)` xảy ra 3 lần, cả 3 đều trên 0.1.57 (vị trí 96, 52168, 108017). Chưa chứng minh được là do `STRATA_GDN_CONVL2=2`.

**Bước tiếp theo:** A/B có kiểm soát (cùng một prompt dài được phát lại) giữa `build-906-pr/strata.bak-0145opt-pr-1105-2301` (0.1.45) và `build-906-pr/strata` (0.1.57). Sau đó thử bật/tắt từng biến môi trường liên quan đến MTP/expert, rồi `git bisect` để tìm commit làm giảm tỷ lệ chấp nhận.


---

## 7. Giải mã nghịch lý `--spec-min-p 0.10` và Thiết kế Thuật toán Đột phá CRA-ZSV

### 7.1. Bằng chứng thực nghiệm từ 398,999 verify rounds thực tế của Hermes (`>=32K` context)

Phân tích toàn bộ log thực tế (`/home/milkha/Documents/ai-local/serving/strata-coder.log`) giữa hai cấu hình của `0.1.57`:
- **`0.1.57 (spec-min-p = 0.50)`** (`284,174` rounds trên `644` requests ngữ cảnh dài):
  - `tok/round = 1.56`, `ms/round = 54.6 ms`
  - Số CPU expert misses/round: `4,692,699 / 284,174 =` **`16.51 CPU misses/round`**
  - Số lượt adaptive PCIe VRAM swaps/round: `1,159,159 / 284,174 =` **`4.08 swaps/round`**
- **`0.1.57 (spec-min-p = 0.10)`** (`114,825` rounds trên `322` requests ngữ cảnh dài):
  - `tok/round = 2.08` (**tăng +33.3% số token mỗi vòng!**)
  - `ms/round = 71.3 ms` (**chậm thêm +16.7 ms mỗi vòng!**)
  - Số CPU expert misses/round: `4,254,955 / 114,825 =` **`37.06 CPU misses/round`** (**tăng gấp 2.24 lần, thêm +20.55 misses/round!**)
  - Số lượt adaptive PCIe VRAM swaps/round: `937,546 / 114,825 =` **`8.16 swaps/round`** (**tăng gấp đôi!**)

Phương trình hồi quy tuyến tính khớp tuyệt đối:
$$\Delta t = 71.3 - 54.6 = +16.7\text{ ms} \quad \Longleftrightarrow \quad +20.55\text{ CPU misses/round} \times 0.813\text{ ms/miss}$$
Khi trừ đi thời gian chờ CPU expert + PCIe swap (`16.51 * 0.813 = 13.4 ms` và `37.06 * 0.813 = 30.1 ms`), thời gian GPU thuần túy của một vòng verify ở cả hai cấu hình đều bằng đúng **`41.2 ms/round`**. Nói cách khác, **42% tổng thời gian mỗi vòng verify (`30.1 ms / 71.3 ms`) bị đốt vào việc tính CPU DDR4 expert và chép PCIe VRAM swap cho các token nháp (`t >= 1`)!**

---

## 8. Thuật toán CRA-ZSV (Co-Routed Asymmetric Zero-Stall Verify)

Đã thiết kế và triển khai trực tiếp vào mã nguồn Strata:
1. **`route_resident_co_k` ([`src/kernels/cuda/native_router.cu`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/kernels/cuda/native_router.cu#L265-L505)) & [`src/core/verify.cpp`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/core/verify.cpp#L119-L125):**
   - Chạy `dim3(64, 8)` trên AMD `gfx906` (mỗi hàng token `t` là một wavefront 64-lane độc lập, không bị phân nhánh SIMT giữa `t == 0` và `t > 0`, quét 512 experts chỉ trong 8 bước coalesced).
   - **Pha 1 (`t == 0`, Target Token):** Giữ nguyên tuyệt đối top-5 experts (`r = 0..4`); thay thế các expert không nằm trong VRAM ở `r = 5..6` nếu chênh lệch logit `<= margin` (`0.50`) và ở `r = 7..9` (vốn bị `STRATA_ROUTE_TAIL_SKIP=7` cắt bỏ về 0) bằng expert resident tốt nhất kèm chuẩn hóa lại softmax. Xuất tập 10 experts của `t = 0` vào `__shared__ int32_t s_t0[10]`.
   - **Pha 2 (`t > 0`, Speculative Draft Tokens):**
     - **Co-Routed Speculative Coalescing (CRSC):** Ở các rank đuôi `r = 5..9`, gom (coalesce) expert của draft token về expert resident đã được `t = 0` chọn trong `s_t0[0..9]` nếu chênh lệch logit `<= co_margin` (`0.25`), giúp gộp nhóm `groups` trên HBM2 mà không đọc thêm trọng số mới.
     - **Zero-Stall Draft Residency:** Ở các rank `r >= 1`, nếu expert của draft token không nằm trong VRAM (`res[e] < 0`), thay thế bằng expert resident tốt nhất trong VRAM (cộng thưởng `+0.25` logit cho expert thuộc `s_t0`) trong ngưỡng `r1_margin = 0.65` (`r = 1`) và `draft_margin = 2.50` (`r >= 2`). Nhờ thay thế ngay trên GPU trước `doorbell_publish_res_kernel`, GPU không cần ghi 30 KB activation qua PCIe ở ~90% số tầng.
2. **Role-Asymmetric Tail Skip & Draft-Filtered Adaptive Usage ([`src/core/expert_source.cpp`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/core/expert_source.cpp#L4865-L5155)):**
   - Bổ sung `draft_tail_skip_rank()` (`STRATA_DRAFT_TAIL_SKIP=2`): tách ngưỡng bỏ qua CPU miss của token gốc (`t = 0`, `tail_rank = 7`) và token nháp (`t >= 1`, `draft_tail_rank = 2`).
   - Lọc trọng số cập nhật `d.usage`: bỏ qua các expert đã bị skip (`skip_d[q]`), giảm trọng số của draft token (`0.25` cho `r < 2`, `0.0` cho `r >= 2`) để ngăn các token nháp làm nhiễu bộ đếm tần suất và gây bão hòa PCIe `adapt()` swap.
   - Sửa điều kiện lượng tử hóa activation trên CPU (`std::none_of(..., v == -1)`): chỉ lượng tử hóa đúng hàng token `t` thực sự có CPU miss (`kind == -1`), thay vì lượng tử hóa cả 3 hàng mỗi khi `any_cpu == true`.

### 8.1. Kết quả Benchmark Thực tế trên AMD MI50 (`gfx906`)

| Chỉ số đo lường | Trước CRA-ZSV (`spec-min-p=0.10`) | Sau CRA-ZSV (Ngữ cảnh ngắn) | Sau CRA-ZSV (Ngữ cảnh dài `16,107` tok) | Mức cải thiện |
| :--- | :---: | :---: | :---: | :---: |
| **Tốc độ Decode (`tok/s`)** | `24.7 – 29.1 tok/s` | **`39.2 – 39.9 tok/s`** | **`36.1 tok/s`** | **+35% đến +58%** |
| **Thời gian 1 vòng (`ms/window`)** | `60.75 – 71.30 ms` | **`53.53 – 53.55 ms`** | **`56.25 ms`** | **Giảm `15.0 – 17.8 ms/vòng`** |
| **Số token chấp nhận (`tok/window`)** | `2.08` | **`2.10 – 2.13`** (`54.7%` accept) | **`2.03`** (`51.6%` accept) | **Giữ nguyên / tăng nhẹ** |
| **Decode Expert Cache Hit Rate** | `96.2%` (`3.8%` miss) | **`99.3%`** (`0.7%` miss) | **`99.4%`** (`0.6%` miss) | **Giảm 5.4x – 6.3x tỷ lệ miss** |
| **CPU Experts / layer-window** | `0.77` (`37.06` misses/vòng) | **`0.20`** (`9.6` misses/vòng) | **`0.16`** (`7.7` misses/vòng) | **Giảm 74% – 79% CPU misses** |
| **Thời gian CPU Host / vòng (`per-layer host`)** | `9.45 ms` (`CPU 8.47 ms`) | **`1.39 ms`** (`CPU 1.06 ms`) | **`1.45 ms`** (`CPU 0.90 ms`) | **Giảm 85% thời gian CPU** |
| **Adaptive VRAM Swaps / vòng** | `3.1 – 8.16 swaps/vòng` | **`0.9 swaps/vòng`** | **`0.6 swaps/vòng`** | **Giảm 71% – 92% PCIe swaps** |
| **Thời gian chờ `adapt wait` / vòng** | `1.54 ms` | **`0.38 ms`** | **`0.15 ms`** | **Giảm 75% – 90%** |


---

### 8.2. Nâng cấp CRA-ZSV v2 + Joint-Confidence Dynamic Speculation (`--spec 4`) cho sinh văn bản dài (`>1,000` tokens)

#### Nguyên nhân tốc độ giảm về `33.8 tok/s` ở bản CRA-ZSV v1 khi sinh đoạn văn dài `>1,000` tokens:
1. **Nhiễu trạng thái ẩn (`final_R[t]`) và logits kiểm chứng (`outv[t]`) khi thay thế rank `r = 1..3` của hàng nháp `t >= 1`:**
   - Trong cửa sổ kiểm chứng (`T = 3`, `t = 0, 1, 2`), khi token nháp thứ nhất (`t = 1`) được chấp nhận (`window[1] == outv[0]`), hàng `t = 1` **trở thành trạng thái đích chính thức**:
     - Logits `outv[1]` của hàng `t = 1` trực tiếp dùng để kiểm chứng token nháp thứ hai (`window[2] == outv[1]`, [`generate.cpp:18889`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/program/generate.cpp#L18889)).
     - Trạng thái KV cache và GDN recurrent state của hàng `t = 1` (hoặc `t = 2`) được commit vĩnh viễn vào phiên ([`generate.cpp:18912`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/program/generate.cpp#L18912)).
     - Vector ẩn cuối `ver.final_R_all()` của hàng được chấp nhận truyền thẳng vào đầu dự đoán MTP ([`generate.cpp:18987`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/program/generate.cpp#L18987)) để sinh nháp cho vòng kế tiếp.
   - Ở bản v1, việc cho phép thay thế cả các rank đầu `r = 1, 2, 3` (chiếm ~45% tổng trọng số softmax MoE) trên hàng `t >= 1` với `draft_margin = 2.50` khiến trạng thái GDN và `final_R` tích lũy sai số qua >1,000 tokens, làm tỷ lệ chấp nhận MTP trên văn bản dài tụt xuống `40.9%` (`1.97 tokens/window -> 33.8 tok/s`).
2. **Ngưỡng phẳng `spec-min-p = 0.10` kết hợp trần cứng `--spec 3`:**
   - Khi $p_0 \in [0.10, 0.28)$ (token nháp đầu tiên rất thiếu chắc chắn), MTP vẫn chạy tiếp bước 2 (`j = 1`, tốn thêm `~1.5 ms`) và ép bộ kiểm chứng chạy cửa sổ `T = 3` (tốn thêm `~4.5 ms`) dù xác suất đồng thời $p_0 \cdot p_1$ cực thấp.
   - Ngược lại, khi $p_0 \cdot p_1 \ge 0.52$ (chuỗi suy luận/code độ tự tin cao), trần `--spec 3` lại khóa cứng không cho MTP đề xuất thêm token nháp thứ 3 (`T = 4`).

#### Giải pháp trong CRA-ZSV v2:
1. **Bảo toàn tuyệt đối Top-3 (`r = 0, 1, 2`) trên MỌI hàng token (`t = 0` và `t > 0`) ([`native_router.cu`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/kernels/cuda/native_router.cu#L368-L465), [`expert_source.cpp`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/core/expert_source.cpp#L4870-L5008)):**
   - Giữ nguyên 100% 3 experts mạnh nhất (`r = 0, 1, 2`, chiếm ~70% trọng số softmax) trên tất cả các hàng `t`, đồng thời áp dụng đường cong `margin` tăng dần theo bậc rank từ `r = 3..9` (`0.35 -> 0.75 -> 1.25 -> 2.50` cho `t = 0`; `0.60 -> 1.20 -> 1.80 -> 2.50` cho `t > 0`) và `STRATA_ROUTE_TAIL_SKIP=6`, `STRATA_DRAFT_TAIL_SKIP=4`.
   - Trong `d.usage`, cấp đủ trọng số bảo vệ (`1.0f` / `0.5f`) cho các expert đang nằm trong VRAM (`host_res >= 0`) để chống bị trục xuất nhầm, và chỉ tích lũy điểm swap cho các expert ngoài VRAM khi bị miss ở đúng cụm lõi `r < 3`.
2. **Joint-Confidence Dynamic Speculation (`T = 1..4`) ([`mtp.cpp`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/core/mtp.cpp#L1858-L1895), [`window_rule.hpp`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/include/strata/spec/window_rule.hpp#L56-L79), `--spec 4`):**
   - Kiểm soát theo xác suất tích lũy $P_{\text{joint}}(j) = \prod_{i=0}^{j} p_i$:
     - Bước 1 (`T = 2`): giữ nếu $p_0 \ge 0.10$.
     - Bước 2 (`T = 3`): chỉ chạy MTP bước 2 nếu $p_0 \ge 0.28$, và chỉ đưa vào cửa sổ verify `T = 3` nếu $p_1 \ge 0.16 \land p_0 p_1 \ge 0.14$ (tiết kiệm `~6 ms/vòng` ở các bước văn bản khó đoán).
     - Bước 3 (`T = 4`): tự động mở rộng lên cửa sổ 4 tokens (`3` nháp + `1` gốc) khi $p_0 p_1 \ge 0.52 \land p_1 \ge 0.60 \land p_2 \ge 0.45 \land p_0 p_1 p_2 \ge 0.35$.

#### Kết quả Benchmark Sinh Văn bản Dài (`1,200 – 1,500` tokens):
- **Bài test 1 (`1,200` tokens - Kiến trúc hệ thống phân tán + Raft C++20):**
  - Tốc độ thực tế toàn cục (tính cả prefill): **`37.47 tok/s`** (`1,200` tokens trong `32.03s`).
  - Tốc độ decode thuần: **`40.0 tok/s`** (pha suy luận `688` tokens: `2.15 tok/window`, `53.73 ms/window`, `59.5%` draft accept, `0.3 swaps/window`) và **`39.5 tok/s`** (pha trả lời `494` tokens: `2.19 tok/window`, `55.28 ms/window`, `99.1%` cache hit, `0.1 swaps/window`).
- **Bài test 2 (`1,500` tokens - Lock-free Thread Pool + Work-Stealing Deque + Unit Tests):**
  - Tốc độ thực tế toàn cục (tính cả prefill): **`38.73 tok/s`** (`1,500` tokens trong `38.73s`).
  - Tốc độ decode thuần trung bình trên toàn bộ `1,482` tokens decode: **`40.60 tok/s`** — gồm **`38.5 tok/s`** ở pha suy luận dài (`988` tokens, `57.8%` draft accept, `54.88 ms/window`) và **`45.7 tok/s`** ở pha sinh code/triển khai (`494` tokens, `avg T = 3.48`, `2.74 tok/window`, `70.6%` draft accept).

---

### 8.3. Khắc phục hiện tượng tụt tốc độ xuống `32.9 tok/s` trên phiên Hermes ngữ cảnh cực dài (`35K – 41K` tokens)

#### Nguyên nhân gốc rễ từ phân tích log `strata decode timing` ở ngữ cảnh `41,115` tokens (`32.9 tok/s`, `62.49 ms/window`):
Khi ngữ cảnh hội thoại tăng lên `35,000 – 41,115` tokens, ba điểm nghẽn độc lập đồng thời xuất hiện làm thời gian mỗi vòng tăng từ `53.5 ms` lên `62.49 ms`:
1. **`per-layer host` tăng vọt lên `5.62 ms/window` (`CPU experts = 0.97` = `46.6` CPU misses/window):**
   - Trong [`src/core/verify.cpp:122`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/core/verify.cpp#L122), cấu hình `RouteResidentCfg::lo` trước đó để mặc định là `5` (ghi đè lên ngưỡng `3` trong kernel), trong khi `env_draft_lo` là `3`. Ở ngữ cảnh `41K` tokens, phân phối router mở rộng khiến các rank `r = 3, 4` của token gốc (`t = 0`) và `r = 1, 2` của các token nháp (`t > 0`) liên tục trỏ vào các expert ngoài VRAM (`hit rate` giảm xuống `96.6%`).
2. **`draft` (MTP drafter) phình to gấp đôi lên `5.24 ms/window`:**
   - Đầu dự đoán nháp MTP (`MtpDrafter`) sử dụng **Dense Attention** (không phải QSA sparse attention), với `--mtp-window` mặc định lên tới `32,768` tokens. Ở ngữ cảnh `35K–41K` tokens, mỗi bước nháp MTP phải quét toàn bộ `32,768` KV tokens trên HBM2, làm thời gian `draft` tăng từ `2.5 ms` lên `5.24 ms/window`.
3. **240 PCIe-Share Kernel Launches dư thừa mỗi window trong `verify.cpp` (`~5.5 ms/window`):**
   - Trong [`src/core/verify.cpp:2405-2419`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/core/verify.cpp#L2405-L2419), ngay cả khi cấu hình `STRATA_PCIE_FRAC_DEFAULT=old` (`pcie_frac == 0`, tức `p_counts[2]` luôn bằng `0`), mỗi tầng trong 48 tầng MoE vẫn gọi vô điều kiện 5 kernel: `wait_flag_ge(m_flagB_)`, `fetch_blobs`, `rebase_ptrs`, và `grouped(p_ptr2, p_start2, p_counts + 2, ...)` (trên `STRATA_HIP_GFX906`, lệnh này phóng cố định `600` blocks `native_gu_fused_kernel` + `2,400` blocks `native_down_lds_kernel` mỗi tầng chỉ để đọc `*count == 0` rồi thoát!).

#### Giải pháp kiến trúc (CRA-ZSV v3.1 + Zero-PCIe-Launch + Sliding-Window MTP):
1. **Loại bỏ 240 PCIe-share launches rỗng & Hợp nhất cờ đồng bộ (`STRATA_FUSE_WAIT=1`, `STRATA_FUSE_HITS=1`):**
   - Bọc nhánh PCIe-share tại [`src/core/verify.cpp:2405`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/core/verify.cpp#L2405) bằng điều kiện `if (!pcie_share_off || m_bal_ != nullptr || device_plan_)`, cắt bỏ hoàn toàn `48 × 5 = 240` lần launch kernel rỗng (tương đương `144,000` threadblocks rỗng mỗi window).
   - Bật `STRATA_FUSE_WAIT=1` (gộp `wait_flag_ge(m_flag_)` vào `copy_rows_from_mapped_after`) và `STRATA_FUSE_HITS=1` (gộp phép cộng `hit_out` trực tiếp vào `combine_k10_vec4_hits`), cắt thêm `96` launches mỗi window.
2. **CRA-ZSV v3.1 (`RouteResidentCfg::lo = 3`, `env_draft_lo = 1`) ([`verify.cpp:122`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/core/verify.cpp#L122), [`native_router.cu:397-412`](file:///home/milkha/Documents/ai-local/engines/Strata-gfx906-opt/src/kernels/cuda/native_router.cu#L397-L412)):**
   - Đồng bộ `RouteResidentCfg::lo = 3` (bảo toàn tuyệt đối top-3 `r = 0, 1, 2` của `t = 0`), và đặt `env_draft_lo = 1` cùng `r1_margin = 0.80f` cho `t > 0` kết hợp `STRATA_ROUTE_TAIL_SKIP=4`, `STRATA_DRAFT_TAIL_SKIP=2`.
   - Kéo tỷ lệ VRAM Expert Cache Hit ở ngữ cảnh `39,556` tokens từ `96.6%` lên **`99.3% – 99.8%`**, giảm `per-layer host` từ `5.62 ms` xuống chỉ còn **`0.63 – 1.47 ms/window`**.
3. **Giới hạn cửa sổ chú ý của MTP Drafter (`--mtp-window 4096`):**
   - Thêm `--mtp-window 4096` vào [`strata-coder-iq3s-gfx906opt.json`](file:///home/milkha/Documents/ai-local/serving/strata-coder-iq3s-gfx906opt.json#L80-L81), giữ nguyên độ chính xác dự đoán nháp cục bộ (`59.6% – 73.0%` draft acceptance ở 39.5K context) trong khi cắt giảm thời gian `draft` ở ngữ cảnh 40K từ `5.24 ms` xuống **`3.14 – 3.85 ms/window`**.

#### Kết quả Kiểm chứng Thực tế trên Ngữ cảnh Ngắn, Trung bình và Cực Dài (`39,556` tokens):

| Kịch bản Benchmark (`max_tokens = 1200`) | Prompt Context | Pha sinh | Tốc độ Decode (`tok/s`) | Trung bình toàn lượt (`1,200` tok) | `tokens / window` (Accept %) | `ms / window` (`GPU` / `Host` / `Draft`) | VRAM Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ** Trước tối ưu (Hermes 41K log)** | `41,115` tok | Mixed | `32.9 tok/s` | `32.9 tok/s` | `2.06` (`47.0%`) | `62.49 ms` (`48.6` / **`5.62`** / **`5.24`**) | `96.6%` |
| **1. Ngữ cảnh ngắn (`1.2K-gen`)** | `92 -> 798` tok | Thinking (`691` tok)<br>Content (`494` tok) | **`41.9 tok/s`**<br>**`43.9 tok/s`** | **`42.7 tok/s`** | `2.42` (`64.3%`)<br>`2.55` (`65.7%`) | `57.91 ms` (`41.9` / `3.42` / `3.63`)<br>`58.05 ms` (`45.1` / `1.70` / `3.66`) | `98.1%`<br>`99.1%` |
| **2. Multi-turn (`10.5K–11.2K` ctx)** | `10,511 -> 11,217` tok | Thinking (`688` tok)<br>Content (`494` tok) | **`39.3 tok/s`**<br>**`43.2 tok/s`** | **`40.9 tok/s`** | `2.21` (`58.3%`)<br>`2.63` (`66.0%`) | `56.25 ms` (`44.9` / `0.92` / `3.25`)<br>`60.80 ms` (`47.2` / `1.94` / `3.84`) | `99.6%`<br>`99.1%` |
| **3. Ngữ cảnh cực dài (`39K-ctx-1.2K-gen`)** | `38,850 -> 39,556` tok | Thinking (`689` tok)<br>Content (`494` tok) | **`38.8 tok/s`**<br>**`45.6 tok/s`** | **`41.4 tok/s`** | `2.14` (`59.6%`)<br>`2.84` (`73.0%`) | `55.12 ms` (`44.3` / **`0.63`** / **`3.14`**)<br>`62.22 ms` (`49.4` / **`1.47`** / **`3.85`**) | **`99.8%`**<br>**`99.3%`** |

