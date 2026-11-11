# gfx906 trên một MI50, pack IQ3_S: phân tích và hướng tối ưu

Tài liệu này tổng hợp việc đọc mã nguồn trên nhánh `v0.1.45-gfx906-opt` (ngày 2026-10-18).
Mọi con số đều ghi rõ đo trên máy nào; những mục **chưa đo** được đánh dấu như vậy, theo đúng
quy tắc trong [AGENTS.md](../AGENTS.md) ("không có tuyên bố mà không có phép đo").

Phạm vi: **một** card gfx906 (Instinct MI50 16 GB hoặc 32 GB), pack **IQ3_S** (`--native`),
decode là mục tiêu chính. Đây là cấu hình mà [AMD_HIP.md](AMD_HIP.md) còn ghi trong mục
"Not done: ... **a single-card run**", nên toàn bộ phần dưới là đề xuất + cách đo, chưa phải kết quả.

---

## 1. Trạng thái hiện tại của nhánh

Ba thay đổi gfx906 đã nằm trong nhánh (`git log`) nhưng **chưa có số đo kèm theo**:

| Commit | Thay đổi | Vị trí | Trạng thái |
|---|---|---|---|
| `10230425` | `dp4a` thật sự bằng `__builtin_amdgcn_udot4` thay vì mô phỏng từng byte | `include/strata/platform/hip_compat/strata_hip.h` | chưa đo |
| `10230425` | `-O3 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false` cho mọi file HIP | `CMakeLists.txt:152` | chưa đo |
| `10230425` | `kExpModeDefault` trên gfx906 đổi từ **7 → 8** (`native_gu_fused_kernel`) | `src/kernels/cuda/iq_kernels.cu:4155-4190` | chưa đo |

Đây là ba thứ đầu tiên cần đo lại, vì chúng đang ở đường mặc định. Chi tiết ở mục 4.

Các fix local đã có từ trước (không đổi hành vi trên card khác, chỉ mở đường build ROCm 6.3.3):
`alignas` trên *type* bị clang 18 từ chối trong device code gfx906 → đổi sang
`__attribute__((aligned(16)))` ở `iq_kernels.cu` (hai mảng `__shared__`) và
`qsa_prompt_attn.cu` (`struct SmemW`); `q36_engine.cu` / `q36ref.cu` và unit NVFP4 W4A4 được
đóng gói `if(NOT STRATA_HIP_GFX906)` (`CMakeLists.txt:709,728,1611`); `src/prefill/moe_f16_kq.cu`
được thêm vào nguồn `strata_mmq` vì `moe_fused_iq.cu` gọi các wrapper của nó
(`CMakeLists.txt:1573`).

## 2. Vì sao gfx906 khác các card AMD khác

- **wave64**, không có WMMA, không có tensor core, không có lệnh SIMD trên byte
  (`AMD_HIP.md:641-646`). Build riêng `-DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906`,
  biên dịch toàn bộ nguồn CUDA thành HIP qua `include/strata/platform/hip_compat/`, một warp CUDA
  ánh xạ vào nửa logical của một wavefront 64 lane. Loại lẫn với `STRATA_ENABLE_HIP`
  (`CMakeLists.txt:64-67`). Card wave32 bị từ chối lúc khởi động (#1728).
- gfx906 báo **compute capability 9.0** qua HIP. Vì thế `dev_info()` trong
  [moe_fused_iq.cu:1258](../src/prefill/moe_fused_iq.cu#L1258) trả về `DevInfo` rỗng trên gfx906:
  các kernel `mma.sync` của đường prompt **không bao giờ** được dùng ở đây. Đường prompt chạy
  MMQ / FP16 GEMM, và QSA prompt attention dùng kernel FP32
  ([qsa_prompt_attn.cu:2262](../src/kernels/cuda/qsa_prompt_attn.cu#L2262)).
  Hệ quả: những tối ưu tensor-core của đường prompt không áp dụng được; muốn nhanh trên gfx906
  phải đi theo MMQ và FP32.
- `--use_fast_math` không bật cho gfx906 (`CMakeLists.txt:515`), `STRATA_MMQ_TAB` tắt
  (`CMakeLists.txt:1618` — gfx906 không biên dịch file `.cu` nào từ thư mục đó, nó dùng glob HIP).
  Bù lại, các instance MMQ cho K-quant được biên dịch thẳng vào `strata_mmq` khi
  `STRATA_MMQ_KQUANTS` (CMakeLists.txt:1586), vì code object HIP nạp khi dùng lần đầu thay vì
  cần `libstrata_kq.so` riêng.

## 3. Ngân sách bộ nhớ: đây là nút thắt số 1 trên một card 16 GB

Số của IQ3_S từ [QUANTS.md](QUANTS.md):

| Hạng mục | Kích thước |
|---|---|
| Routed experts (UD-IQ4_XS gate/up, IQ3_S ở các layer còn lại) | **55.4 GiB** |
| Mỗi token (10 expert × 48 layer) | 1.16 GB |
| Dense side GPU đọc một lần mỗi verify window | ≈ **4.8 GB** |
| PLE table nếu IQ4_NL (IQ4_XS / Q4_K_XL) | 26.8 GiB |
| PLE table nếu Q8_0 | 50.7 GiB |

Một MI50 16 GB **không** chứa được dense side + KV + PLE table cùng lúc, nói gì tới 55.4 GiB experts.
Vậy experts ở RAM hệ thống, đi qua expert cache và KV streaming. Điều đó định hình mọi thứ:

- **Số slot expert cache quyết định tốc độ decode.** Hai báo cáo cộng đồng cùng pack (IQ2_XS) khác
  nhau đúng ở chỗ đó:
  - MI50 **32 GB** single, i5-12400F, 64 GB RAM, `--expert-cache auto` → 19,304 slot,
    hit rate 0.97–0.995, decode **36.0 / 34.0 / 33.5 tok/s** ở 4K / 32K / 128K, prompt ~330:
    [bench/results/2026-10-04-community-mi50](../bench/results/2026-10-04-community-mi50/README.md)
  - Radeon Pro VII **16 GB** single, Xeon E5-2666 v3, 125.7 GiB RAM, thêm
    `--vram-reserve-mib 1024`, băng thông host→device đo được **13.7–13.8 GB/s**
    → decode **24.2 / 24.0 / 23.7 tok/s**, prompt 350 / 339 / 284:
    [bench/results/2026-10-09-community-radeon-pro-vii-16gb](../bench/results/2026-10-09-community-radeon-pro-vii-16gb/README.md)

  Cùng kernel gfx906, cùng pack: card 16 GB chậm hơn **~33%** ở decode, và báo cáo giải thích bằng
  số slot expert cache ít hơn. Với IQ3_S, experts lớn hơn IQ2_XS, nên áp lực này còn lớn hơn.
- **Luật cache file tier.** gfx906 giữ luật cũ theo mặc định (`STRATA_FILE_CACHE_OLD_RULE`,
  xem bảng switch trong `AMD_HIP.md`). #1752 là lý do: trên MI50, luật mới gây HMM stall và
  prompt rơi từ 46 lên 360 tok/s khi đổi luật. **Không** bật luật mới trên card này khi chưa đo.
- **PLE table nên là IQ4_NL/IQ4_XS, không phải Q8_0** (26.8 GiB thay vì 50.7 GiB). Trên một card
  16 GB với RAM có hạn, 24 GiB chênh lệch đó mua thêm slot expert cache.

### Config khởi điểm để đo (suy ra từ hai báo cáo trên, chưa kiểm chứng với IQ3_S)

```sh
HSA_OVERRIDE_GFX_VERSION=9.0.6   # chỉ khi runtime không nhận card
strata serve -m <Qwen3.8-Flash-Next-IQ3_S.gguf> --native \
  --kv int8 --kv-resident 32768 \
  --vram-reserve-mib 1024 \
  --prefill 2048 --spec 3 --spec-min-p 0.9 \
  --expert-cache auto
```

So với bản 32 GB IQ2_XS thì khác ở: giữ `--vram-reserve-mib 1024` (bắt buộc trên 16 GB, theo
báo cáo Pro VII), và `--prefill 2048` chứ không 2048→4096 (IQ3_S lớn hơn, chunk prompt chèn vào
VRAM đang chật). `--spec` bắt đầu ở 3 như bản 32 GB; bản 2×MI50 dùng 4 nhưng có hai card.

## 3b. "6B active" là con số FLOP, không phải con số byte

Mô hình 512 expert/layer mà chỉ 10 expert chạy mỗi token làm **FLOP** giảm ~50 lần, nhưng
**không** làm giảm **số byte phải đọc** đi tương ứng, và decode ở T nhỏ bị ràng buộc bởi số byte:

| Phần | Đọc bao nhiêu | Có giảm nhờ sparsity không |
|---|---|---|
| Routed experts (10 × 48 layer) | IQ3_S: 1.16 GB/token | Có |
| Dense side: attention, SSM, hyper-connection, shared expert, PLE key/value, embedding | **≈4.8 GB cho mỗi verify window, như nhau ở cả 5 file** (QUANTS.md:28-30) | **Không** |

Nghĩa là ở 2 token/window, chi phí cố định (~2.4 GB/token) **gấp đôi** chi phí expert (1.16 GB/token).
Sparsity chỉ tác động vào phần nhỏ hơn.

### Phép so sánh trần (tôi tính từ các số đã công bố, chưa phải phép đo mới)

Lấy breakdown đã đo trên R9700 + UD-Q4_K_XL làm chuẩn để kiểm tra cách tính: window 34.3 ms,
bound băng thông ~12.5 ms → **36%** (khớp đúng con số "35-40% của bound" trong QUANTS.md:113).

Cùng cách tính cho các cấu hình khác:

| Cấu hình | Byte/token | Bandwidth | Bound | Đo được | % của bound |
|---|---|---|---|---|---|
| R9700, IQ3_S | ~3.6 GB | 574 GB/s | ~6.2 ms | 13.3 ms (75 tok/s) | **~47%** |
| MI50 32 GB single, IQ2_XS | ~2.3 GB | 1,024 GB/s | ~2.3 ms | 27.8 ms (36 tok/s) | **~8%** |
| Pro VII 16 GB single, IQ2_XS | ~2.3 GB | 512 GB/s | ~4.5 ms | 41.3 ms (24.2 tok/s) | **~11%** |

Kết luận quan trọng nhất của cả tài liệu này: **MI50 không chạy sát trần băng thông của nó.**
Card 32 GB có HBM2 1 TB/s — nhanh hơn cả R9700 — nhưng chỉ đạt 8% bound, trong khi card hiện đại
đạt 47%. Chênh lệch đó không phải bộ nhớ, mà là **_throughput lệnh và độ trễ kernel của ISA 2018_**:

1. **Không tensor core, không WMMA.** Mọi GEMM chạy trên FMA vô hướng.
2. **Warp CUDA là một nửa *logical* của wavefront 64 lane**
   ([strata_hip.h:7-11](../include/strata/platform/hip_compat/strata_hip.h#L7)). **Không** có chuyện
   một nửa lane rảnh: block size trên đường decode là 256 và 512
   ([fused_gr.cu:39](../src/kernels/cuda/fused_gr.cu#L39), [router_top10.cu:65](../src/kernels/cuda/router_top10.cu#L65)),
   bội số của 64, nên cả 64 lane đều làm việc. Cái giá thật của việc giữ shape 32 lane là ở chỗ khác:
   mỗi cây giảm tổng chạy 5 lần `__shfl_*_sync` width 32 thay vì 1 butterfly 64 lane
   ([strata_hip.h:197-205](../include/strata/platform/hip_compat/strata_hip.h#L197)), và mỗi
   `__syncwarp` là một **wavefront barrier** thật (fence + `wave_barrier` + fence,
   [strata_hip.h:216-221](../include/strata/platform/hip_compat/strata_hip.h#L216)) chặn cả 64 lane
   thay vì 32. Bằng chứng width 64 đáng: `reduce12_gfx906` báo "16 shuffle exchanges instead of 60",
   `native_mmvq.cu` dùng butterfly 64 lane, và [iq_kernels.cu:3596](../src/kernels/cuda/iq_kernels.cu#L3596)
   đã gọi `__shfl_xor(s, o, 64)`. Số kernel có layout wave64 riêng vẫn chỉ đếm trên đầu ngón tay:
   MMVQ, grouped experts, router top-10, GR, QSA attention.
3. **IQ dequant không có SIMD byte** — bảng IQ4 đi qua chuỗi `v_perm_b32`, `dp4a` phải mô phỏng
   (giờ là `udot4`). Vòng trong decode bị ràng buộc bởi số lệnh, không bởi số byte.
4. **Nhiều kernel nhỏ mỗi window** không lấp được 60 CU. Ví dụ đã được sửa trong chính nhánh này:
   [fused_gr.cu:1512-1514](../src/kernels/cuda/fused_gr.cu#L1512) — grid cũ "để lại phần lớn card
   idle và đọc ma trận 6.5 MB ở ~100 GB/s" thay vì ~500.

Vậy room thuật toán **có thật và lớn (~3-4 lần)**, nhưng nằm ở *số kernel và layout lane*, không
nằm ở công thức toán.

### Prefill lại là chuyện khác: nó đã gần trần FLOP

MI50 (Vega 20) có ~7.4 TFLOPS FP32. Prefill 6B active cần `2 × 6e9 × T` FLOP:

- prompt ~330-350 tok/s đo trên MI50/Pro VII đơn ⇒ ~4.2-4.6 TFLOPS hiệu dụng ⇒ **~57-62% đỉnh FP32**.

Đó là hiệu quả cao. **Không còn room thuật toán đáng kể cho prefill trên gfx906** nếu vẫn nhân FP32.
Cần đổi *số học*, không phải đổi kernel: FP16 packed (Vega 20 có cổng nửa tốc độ) hoặc int8. Nhưng
lưu ý: gfx906 không có `V_DOT`, `udot4` phải xuống thành chuỗi `v_macx_u8`/`v_madak_u8` — **tốc độ
8-bit trên gfx906 ngang FP32, không nhanh hơn 4 lần**. Trên card này, lượng tử hoá mua **byte**,
không bao giờ mua **FLOP**. Đó là lý do `--use_fast_math` bị tắt (`CMakeLists.txt:515`) và đường
prompt ở lại FP16/FP32.

## 4. Bốn hướng tối ưu, theo thứ tự đáng làm

### 4.1. Đo lại `STRATA_EXP_MODE` cho IQ3_S trên gfx906 — ưu tiên cao nhất

`exp_mode()` ([iq_kernels.cu:4155-4190](../src/kernels/cuda/iq_kernels.cu#L4155)) hiện trả về
**8** cho gfx906. Tiền lệ trong nhánh CUDA: mode 8 được chọn làm mặc định cho sm_60 (P100),
sm_61 (P40) và sm_70, kèm số đo — IQ3_S, lời gọi 18 expert tại 2 token: **390 → 255 µs trên P40**.
Đó chính xác là hình dạng công việc của gfx906 (không tensor core, decode T nhỏ, IQ format).

Mode 8 là `native_gu_fused_kernel` ([iq_kernels.cu:4066-4071](../src/kernels/cuda/iq_kernels.cu#L4066)):
gate/up của mode 7 **cộng thêm** SwiGLU và lượng tử hoá q8_1 ngay trong epilogue; một block lấy `h`
rows `[r0, r0+32)`, 64 weight rows, **cùng tổng per-(row, entry) như mode 7** — tức đổi này phải
bit-exact so với mode 7, không phải xấp xỉ. Launch với `TG` 16/18/21/22/23
([iq_kernels.cu:4343-4347](../src/kernels/cuda/iq_kernels.cu#L4343)).

Cách đo (không cần model cho bước đầu):

```sh
# 1) bit-exactness: mode 8 phải khớp mode 7 trên cùng dữ liệu tổng hợp
STRATA_EXP_MODE=7 build-906/native_expert_bench ...
STRATA_EXP_MODE=8 build-906/native_expert_bench ...
# CMakeLists.txt:2009 đã có sẵn bài parity cho cặp mode 7 + STRATA_HIP_SWIGLU_FUSED=1
ctest --test-dir build-906 -R 'native_grouped'

# 2) tốc độ theo từng format, trên dòng GGUF thật
for m in 2 7 8; do STRATA_EXP_MODE=$m build-906/native_expert_bench --format iq3_s ...; done
```

Nếu mode 8 không thắng trên gfx906, **đổi mặc định về 7** và để mode 8 sau một switch tường minh —
theo luật "đổi mặc định chỉ ở nơi đã đo là nhanh hơn, kèm số". Cần đặc biệt chú ý `TG`: các giá trị
16/18/21/22/23 được chọn theo occupancy của card khác; 60 CU × 64 KiB LDS của gfx906 có thể cho
occupancy khác. Chạy `rocm-profiler` hoặc `HIP_VISIBLE_DEVICES=1` + `rocminfo` để đọc
VGPR/LDS thực tế của kernel (bài swizzle ở mục 4.3 đã làm đúng kiểu đó: VGPR 41 → 42).

### 4.1b. Số đo thật đầu tiên, lấy trên máy này ngày 18/10

Máy đo: **một MI50 32 GB, gfx906, 60 CU**, ROCm 6.3.3, `build-906`
(`-DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906`, Release). Build sạch 100%.

Parity (5/5 pass, mỗi bài chạy trong 4 GB VRAM trống — server production vẫn đang chạy):

```
mmvq_multi_parity .............. Passed   0.67 s
iq_multi_parity ................ Passed   1.59 s
native_grouped_parity .......... Passed   1.93 s
native_grouped_fused_parity .... Passed   1.92 s   # mode 7 + STRATA_HIP_SWIGLU_FUSED=1
gfx906_gdn_rec_head ............ Passed   3.69 s
```

`native_expert_bench <shard> <layer> <groups> <tokens> <ref> <mode> <iters>`, trên đúng dòng GGUF
IQ3_S của máy này, mọi cặp đều **bitwise equal**:

| lớp | G×T | mode 7 | mode 8 | mode 8 so với 7 |
|---|---|---|---|---|
| 0 | 4×8 | 170.0 µs | 162.3 µs | −4.5% |
| 5 | 10×1 | 101.1 µs | 89.5 µs | **−11.5%** |
| 5 | 10×4 | 216.9 µs | 190.6 µs | **−12.1%** |
| 5 | 40×8 | 1383.8 µs | 1423.7 µs | +2.9% |

Mode 2 (chậm nhất, chuẩn để so): 406.7 µs ở 4×8 lớp 0 — tức **2.4 lần chậm hơn mode 8**; mode 0
(ref) cũng chậm hơn mode 7/8 ở mọi shape. Kết luận cho nhánh: **mode 8 là mặc định đúng cho decode
trên gfx906, và lần này có số**; shape lớn (40×8, gần prefill) thì mode 7 nhỉnh hơn 2.9%.

Băng thông kernel đọc được từ cùng phép đo: **169–251 GB/s ở shape decode (10×1)**, rớt về
48–62 GB/s ở 40×8, trên card có HBM2 ~1 TB/s. Đây là bằng chứng trực tiếp cho mục 3b: kernel expert
decode chạy ở ~20% trần băng thông của chính nó.

Còn một điều các con số lộ ra mà trước đó tài liệu này giả định sai: các lớp trong pack này
**không cùng một format**. `native_expert_bench` báo `iq3_xxs` ở lớp 0 và `iq2_s` ở lớp 5 (cả hai
`/iq4_nl` cho down). Tức "IQ3_S 55.4 GiB" là tên gói, không phải một format duy nhất — mọi tính toán
byte theo format ở mục 3b cần đọc lại theo từng lớp.

### 4.2. Hai switch attention gfx906: có số đo rồi, đang tắt mặc định

| Switch | Số đo có sẵn | Nguồn |
|---|---|---|
| `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` | **+14.65%** trên thành phần INT8 batched-QSA, **+1.19%** prompt ở model-level 65K, **+0.42%** generation; VGPR 41→42 | [bench/results/2026-10-08-gfx906-qsa-query-swizzle](../bench/results/2026-10-08-gfx906-qsa-query-swizzle/README.md) |
| `STRATA_GFX906_ATTN_REDUCE12=1` | giảm 16 lần shuffle exchange thay vì 60 (`reduce12_gfx906`, [qsa_decode_attn.cu:93](../src/kernels/cuda/qsa_decode_attn.cu#L93)) | PR #1718, mặc định tắt |

Cả hai chỉ được biên dịch vào build gfx906, đều off. Đây là ứng viên "đổi mặc định" hợp lệ nhất
trong toàn bộ danh sách, **nhưng** số đo trên là cho INT8 pool và Coder IQ1_M, không phải IQ3_S
trên một card. Việc cần làm: chạy lại đúng probe + guard có sẵn

```sh
build-906/gfx906_qsa_query_swizzle_probe
python3 bench/test_gfx906_qsa_query_swizzle_guard.py
```

trên cấu hình single-MI50 + IQ3_S, cả 4K/32K/128K, và đọc guard ở
[qsa_decode_attn.cu:824](../src/kernels/cuda/qsa_decode_attn.cu#L824)
(`gfx_arch_is(prop.gcnArchName, "gfx906") && prop.warpSize == 64`) — một MI50 luôn qua guard này,
nên không có lý do kỹ thuật nào để tắt ngoài việc chưa đo.

Lưu ý: swizzle chỉ áp dụng cho **INT8 pools** trong `qsa_decode_attn_batch`
([qsa_decode_attn.cu:1080-1143](../src/kernels/cuda/qsa_decode_attn.cu#L1080)). Trên card 16 GB,
`--kv int8` là mặc định nên đã bật điều kiện này.

### 4.3. Những switch chưa bao giờ đo trên gfx906

| Switch | Đã đo ở đâu | Ghi chú cho gfx906 |
|---|---|---|
| `STRATA_SELECT_SIMT=1` (bộ chấm điểm QSA FP32 tiled) | gfx103x: +7.4% ở prompt 128K | gfx906 không có tensor core, nên đường SIMT FP32 là đường *tự nhiên* của nó; rất có thể nên là mặc định. Chưa đo. |
| `STRATA_GR_SPLIT=1` (`gr_down_multi` split theo K, 8 rows × 5 slice 2048) | AMD, ~3%/window | tổng theo thứ tự khác → **không** bit-exact, nên off. Chỉ bật nếu chấp nhận khác bit. |
| `STRATA_ROUTE_PER_TOKEN=1` | khôi phục layout CUDA | verify-window routing tốn ~3 ms của window ~55 ms. A/B rẻ. |
| `STRATA_MMVQ_WAVE=0` | — | layout wave64 (butterfly 64 lane, không LDS, [native_mmvq.cu:1811](../src/kernels/cuda/native_mmvq.cu#L1811)) thắng ở báo cáo 2×MI50; trên một card với cache miss cao hơn có thể khác. |
| `STRATA_GR_FAST=0` | 60 CU gfx906, [fused_gr.cu:1512-1514](../src/kernels/cuda/fused_gr.cu#L1512) | fix grid để lại phần lớn card idle và đọc ma trận 6.5 MB ở ~100 GB/s. Nên giữ bật. |
| `STRATA_MMQ_KQ=1` / `STRATA_MMQ_KQUANTS` | — | vì `dev_info()` rỗng, đường prompt *phụ thuộc* MMQ. Kiểm tra instance IQ3_S có mặt trong `strata_mmq` ([CMakeLists.txt:1556](../CMakeLists.txt#L1556) liệt kê `iq3_s`). |

### 4.4. Ba thay đổi chưa đo ở mục 1: tách riêng ra để quy lỗi

`udot4`, cờ inline LLVM và mode 8 đến trong **cùng một commit**. Khi có hiện tượng hồi quy, không
biết thủ phạm nào. Đề xuất: build ba bản trong `build-906/` — baseline (`STRATA_EXP_MODE=7`,
không cờ inline, dp4a mô phỏng), chỉ-udot4, chỉ-cờ-inline — rồi so bằng `native_expert_bench` và
`mmvq_il_parity` / `iq_multi_parity` / `mmvq_multi_parity`. `mmvq_multi_parity_cuda_layout`
(`CMakeLists.txt:914`, chạy với `STRATA_MMVQ_WAVE=0`) là negative control có sẵn cho phần này.

### 4.5. Ba hướng thuật toán thật sự (không phải đổi switch)

Vì mục 3b chỉ ra chỗ hở nằm ở *số kernel + số lệnh mỗi lane + byte dense side*, công việc có giá trị
nhất là ở đó:

1. **Rút ngắn chuỗi shuffle / barrier của các kernel còn mang shape 32 lane.** Không có lane nào
   rảnh (mục 3b.2), nên trần lợi nhuận **không phải 2x mỗi kernel**; cái được là số lệnh mỗi lane:
   một cây giảm 32 lane tốn ~5 lần shuffle + một barrier toàn wavefront, trong khi butterfly 64 lane
   tốn 1 lần. `reduce12_gfx906` đo được "16 shuffle exchanges instead of 60" cho đúng loại việc này.
   Việc: liệt kê kernel theo thời gian bằng `rocprof --stats` trên một decode window, rồi đổi cây
   giảm sang butterfly 64 lane ở đầu danh sách — chỉ bit-exact khi thứ tự giảm không đổi, và phải
   chứng minh điều đó từng kernel (64 lane gộp hai half 32 lane không cùng thứ tự kết hợp).
2. **Gộp các kernel nhỏ trong window.** Window decode có hàng trăm kernel nhỏ; độ trễ khởi tạo và
   barrier của chúng là phần lớn khoảng cách tới bound (QUANTS.md:113-115 ghi rõ "the per-window
   latency of the small kernels and the draft"). `native_gu_fused_kernel` (mode 8) chính là nước đi
   này cho gate/up + SwiGLU + q8_1; còn norm, hyper-connection và head chưa được gộp trên gfx906.
   `STRATA_HC_UPMIX=1` (gộp up projection với `gr_mix_r`) đã đo +0.6%/+0.4% prompt nhưng gate ở
   sm_80+ ([DETAILS.md:2234](DETAILS.md#L2234)) — gfx906 không tới được nhánh đó.
3. **Giảm byte dense side, không phải byte expert.** Dense side 4.8 GB/window là phần lớn công mỗi
   token và sparsity không chạm tới nó. `STRATA_HC_REQ8=1` lượng tử hoá hyper-connection thành int8
   + một scale mỗi 32, **đổi thẳng byte thành slot expert cache**, đã đo decode −0.2% / **+2.2%**
   với KL 0.0018 ([DETAILS.md:2236](DETAILS.md#L2236)) — chưa từng chạy trên gfx906, và trên card
   16 GB nơi slot là nút thắt thì giá trị của nó có thể lớn hơn nhiều con số 2.2%.

Thứ tự này ngược với cảm giác thông thường (tối ưu kernel expert trước): expert chỉ là
1.16/3.56 GB, còn 2.4 GB/token là dense side mà không sparsity nào giảm được.

## 5. Quy trình đo bắt buộc trước khi đổi bất kỳ mặc định nào

Theo `AMD_HIP.md`: mỗi switch phải được kiểm **bitwise** với layout nó thay thế, qua ba lớp.

```sh
# lớp 1 — parity trên dữ liệu tổng hợp, không cần model, không cần GPU ngốn nhiều
ctest --test-dir build-906 -R 'iq_multi_parity|mmvq_multi_parity|mmvq_il_parity|native_grouped|shared_expert_parity'

# lớp 2 — kernel thật trên dòng GGUF thật
build-906/native_expert_bench          # mode nọ vs mode kia
build-906/qsa_select_bench             # SELECT_SIMT

# lớp 3 — văn bản greedy của một request cố định, chạy hai lần để chắc chắn ổn định
```

Profile: `STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`. Trên R9700 với UD-Q4_K_XL, window
2.0 token là 34.3 ms và bound băng thông là ~12.5 ms; kernel format đạt ~70% VRAM bandwidth,
CPU pool 84–97% RAM bandwidth, head 83% (QUANTS.md:110-116). Trên MI50 16 GB, phần "experts trong
VRAM" biến thành "expert cache + PCIe 13.7 GB/s", nên khi đọc profile cần chia riêng:

- thời gian trong kernel (GPU bandwidth),
- thời gian chờ upload expert qua PCIe,
- hit rate của expert cache (báo cáo 32 GB: 0.97–0.995).

Nếu hit rate dưới ~0.95 trên IQ3_S, tối ưu kernel không phải nút thắt — tăng slot hoặc đổi
PLE table mới là nút thắt.

## 6. Thứ tự việc nên làm

1. Build `build-906` sạch, chạy toàn bộ `ctest -R parity` để chốt nền tảng bit-exact.
2. Chạy config mục 3 với **IQ2_XS** trên đúng card định dùng, so với hai báo cáo cộng đồng để
   chắc môi trường (ROCm, `HSA_OVERRIDE_GFX_VERSION`, RAM) cho ra số giống hạng.
3. Đổi sang **IQ3_S**, ghi lại decode/prompt 4K/32K/128K, hit rate và số slot expert cache.
   Đây là con số gốc cho mọi so sánh sau.
4. `native_expert_bench` cho mode 2 / 7 / 8 với IQ3_S; nếu mode 8 không thắng, đổi mặc định
   `exp_mode()` về 7 ngay (đây là mặc định chưa được chứng minh).
5. Tách `udot4` và cờ inline LLVM thành hai phép đo riêng; giữ cái nào có số.
6. Bật `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` và `STRATA_GFX906_ATTN_REDUCE12=1` trên IQ3_S;
   nếu lặp lại được +1.19% prompt / +0.42% generation thì đề xuất đổi mặc định (PR riêng,
   mỗi switch một PR).
7. A/B `STRATA_SELECT_SIMT`, `STRATA_ROUTE_PER_TOKEN`, `STRATA_MMVQ_WAVE` — mỗi cái một phép đo.
8. Chỉ sau khi có số: viết một PR cho một thay đổi, kèm cách kiểm và số đo trên cấu hình
   single-MI50 + IQ3_S, và nói rõ cái gì chưa kiểm (MI60, Radeon VII, Windows, `--vision`).

## 6b. Đối chiếu kế hoạch "prefill 1,000 tok/s" (nhận ngày 19/10)

Một bản kế hoạch được đưa ra với mục tiêu **≥1,000 tok/s prefill** cho Qwen3.8-Flash-Next Coder
IQ3_S trên một MI50, gồm ba phần: kernel Flash-Attention FP16 wave64, kernel MoE grouped dùng
`v_dot4_i32_i8`, và một ring buffer prefetch hai stream. Từng con số của kế hoạch được đối chiếu
dưới đây. **Bốn con số nền của nó sai**; hai phần còn lại thì mã nguồn đã có sẵn.

| Kế hoạch nói | Mã nguồn / số đo nói | Nguồn |
|---|---|---|
| MI50: **52.2 TOPS INT8**, 26.8 TFLOPS FP16 | gfx906 **không có `V_DOT`**; `udot4` xuống chuỗi `v_macx_u8`/`v_madak_u8`, nên **int8 ngang FP32**. Đỉnh FP32 của card ~7.4 TFLOPS, FP16 packed chạy trên cổng nửa tốc độ. Kiểu tính 4 byte/lane chỉ đúng với kiến trúc có `V_DOT` (CDNA1 trở lên) | mục 3b ở trên; `AMD_HIP.md:641-646` |
| "28.08 GB experts" phải qua PCIe mỗi pass | Routed experts của IQ3_S là **55.4 GiB** (≈1.16 GB/token), không có file nào trong pack ở cỡ 28 GB. IQ2_XS — pack nhẹ hơn — cũng đã lớn hơn con số đó | [QUANTS.md](QUANTS.md), bảng mục 3 |
| Coder có **512 expert** | Coder giữ **256 trong 512** expert, chọn theo dữ liệu code | [MODELS.md:20](MODELS.md#L20) |
| 7.6 GFLOPs/token ⇒ 1,000 tok/s ⇒ **7.6 TFLOPS** | Đã vượt **103% đỉnh FP32** của card. Với con số FLOP đúng hơn ở mục 3b (~12 GFLOP/token), 1,000 tok/s cần **~12 TFLOPS = 1.6× đỉnh FP32** | mục 3b; phép tự kiểm: 330-350 tok/s × 12 GFLOP ≈ 4.2-4.6 TFLOPS, khớp "~57-62% đỉnh" |

Hai phần còn lại không phải việc chưa làm:

- **Phần attention.** `return false` ở
  [qsa_prompt_attn.cu:2262](../src/kernels/cuda/qsa_prompt_attn.cu#L2262) từ chối **đường int8-KV
  tensor-core / WMMA**, không phải "gfx906 không có kernel attention". Ở
  [prefill.cpp:3265](../src/prefill/prefill.cpp#L3265), khi hàm đó trả `false`, prefill rơi ngay về
  `qsa_decode_attn_batch` — kernel wave64 đã chạy. Và attention ở đây là **sparse**: bề rộng lựa chọn
  là `min(n_kv, idx_top_k + idx_block - 1)` = **2,051 cell tối đa, không đổi theo độ dài prompt**
  ([qsa.hpp:127-136](../include/strata/kernels/qsa.hpp#L127)). Với `n_head` 24 và `head_dim` 256
  ([geometry.hpp:39-41](../include/strata/core/geometry.hpp#L39)), phần QK+AV của một query token là
  `2 × 2 × 2051 × 256 × 24` ≈ **50 MFLOP/token**, tức **dưới 1%** tổng FLOP mỗi token (chưa đo trực
  tiếp; tính từ shape mặc định). Một kernel nhanh gấp 4-6 lần trên thành phần đó mua **dưới 1%**
  end-to-end. Tiêu chí "attention cho prompt 10k ≤3 s" tương ứng ~0.5 TFLOP công việc, đã đạt dư
  hàng chục lần. Đây là lý do kế hoạch đó không thể tới 1,000 tok/s dù chạy đúng: nó tối ưu chỗ rẻ.
- **Phần ring buffer prefetch.** Đã có sẵn và đã được chỉnh theo byte: `RING_MAX = 1024`
  ([prefill.cpp:171](../src/prefill/prefill.cpp#L171)), `fused_ring()` (dòng 255), `ring_cap()` (274),
  `ring_bytes()` (279, 1024 slot cho đường fused non-native, nếu không thì 384),
  `ring_slots()` (307, có override `STRATA_PREFILL_RING`), `stage_slots()` (332), và nhánh rút về 96
  slot khi host copy chiếm ≥90%. Việc còn lại là **đo và đổi số slot**, không phải viết machinery mới.
- **Phần MoE grouped / MMQ.** Đường fused native đã tồn tại
  (`native_supported()` ở [moe_fused_iq.cu:1386](../src/prefill/moe_fused_iq.cu#L1386)); sở dĩ nó tắt
  trên gfx906 là vì `dev_info()` trả `DevInfo` rỗng (HIP báo cc 9.0), không phải chỉ vì thiếu WMMA
  (mục 2). Và chính nhánh này đang có `src/prefill/moe_mmq_gfx906.cu` +
  `include/strata/prefill/moe_mmq_gfx906.hpp` chưa commit, nối vào `Gemm::native()` qua
  `STRATA_GFX906_MMQ_MAX_T` — tức hướng đó đã được bắt đầu, chỉ chưa có số.

### Số đo ngày 19/10: `moe_mmq_gfx906` chậm hơn đường FP16 ở mọi `T`

Đo trên máy này (một MI50 32 GB, gfx906, 60 CU, ROCm 6.3.3, `build-906`), median of 20. Weights lượng
tử hoá theo `quantize_row_iq3_s_ref` / `quantize_row_iq4_nl_ref`, activation FP16 ngẫu nhiên; cột phải
là đường prefill gfx906 đang dùng (`dequant_f16` + `gemm_f16`).

| Tensor | `T` | `moe_mmq_gfx906` (`udot4`) | `dequant_f16` + `gemm_f16` | Chậm hơn |
|---|---|---|---|---|
| iq3_s gate/up (N 1280, K 2560) | 128 | 1.359 ms | 0.214 ms | **6.3×** |
| | 256 | 2.861 ms | 0.404 ms | 7.1× |
| | 512 | 5.910 ms | 0.530 ms | **11.2×** |
| | 1024 | 12.131 ms | 0.978 ms | 12.4× |
| iq4_nl down (N 2560, K 640) | 128 | 0.391 ms | 0.103 ms | 3.8× |
| | 256 | 0.756 ms | 0.150 ms | 5.0× |
| | 512 | 1.671 ms | 0.259 ms | 6.5× |
| | 1024 | 3.411 ms | 0.456 ms | 7.5× |

Sai số so với reference FP16: relative L2 **0.0037-0.0038** (0.37-0.38%), max abs diff 0.021-0.051,
không NaN/Inf — tức đúng dải nhiễu của việc lượng tử hoá activation sang q8_1, đường đó *đúng*, chỉ
là **chậm**. Kernel hiện giảm tổng bằng tree reduction ở mức wave, chưa có LDS tile 2D.

Ba kết luận:

1. **Không có `T` nào MMQ thắng**, nên không có ngưỡng để đặt. Default của `STRATA_GFX906_MMQ_MAX_T`
   trong [gemm.cu:1273](../src/prefill/gemm.cu#L1273) đã đổi **512 → 0** (tắt), kèm ghi chú số đo; vẫn
   đặt `=128` trở lên để bench khi làm tiếp kernel đó. Đây là fix một dòng, không phải tối ưu.
2. Cột FP16 cho ra con số quan trọng nhất: **6.86-7.35 TFLOPS ở `T` ≥ 512**, bằng **93-99% đỉnh FP32
   7.4 TFLOPS** của card. FP16 trên gfx906 chạy *cùng nhịp* FP32 chứ không nhanh gấp đôi, khớp mục 3b
   ("cổng nửa tốc độ" nghĩa là không được gì thêm, không phải được 2×).
3. Vì thế trần của prefill không còn là ẩn số: GEMM đã chạm 93-99% đỉnh, còn prefill end-to-end ở
   4.2-4.6 TFLOPS (mục 3b). Khoảng cách **1.5-1.7×** chính là toàn bộ room, và nó khớp độc lập với
   mục tiêu 500-600 tok/s đặt ở mục này.

### Mục tiêu thay thế, kèm điều kiện

Mục 3b đã đo prefill ở **57-62% đỉnh FP32** (~4.2-4.6 TFLOPS hiệu dụng ở 330-350 tok/s). Phép đo
19/10 dưới đây đóng đường int8 và cho biết trần thật: GEMM FP16 cô lập đạt **6.86-7.35 TFLOPS**, tức
**93-99% đỉnh** của card. Room còn lại của prefill vì thế là **1.5-1.7×**, nằm ở *thời gian ngoài
GEMM*, không ở công thức nhân — khác với giả định ban đầu của mục 3b rằng phải đổi sang FP16 packed
mới có tốc độ. Cụ thể, thứ tự phải đo:

1. ~~Đo `moe_mmq_gfx906` so với đường FP16, quét `STRATA_GFX906_MMQ_MAX_T`.~~ **Đã đo 19/10** (bảng
   trên): MMQ chậm hơn 3.8-12.4× ở mọi `T`, default đã tắt (`MAX_T=0`). Hướng int8 đóng lại.
2. **Việc còn lại là tìm chỗ hở giữa 4.4 và 7.0 TFLOPS, không phải viết GEMM mới.** Đường FP16 hiện tại
   đã tới 93-99% đỉnh ở `T` ≥ 512 trong phép đo cô lập, nên một kernel `v_pk_fma_f16` tự viết chỉ ăn
   được phần chênh nhỏ, và FP16 **không** bit-exact với FP32 (phải kèm KL hoặc parity có ngưỡng,
   switch opt-in). Bước có giá trị hơn: `rocprof --stats` trên một prompt 4K/32K để xem bao nhiêu thời
   gian nằm ngoài GEMM — dequant `dequant_f16` chạy trước mỗi GEMM là một ứng viên rõ ràng, vì nó đọc
   và viết lại toàn bộ weight ra FP16 cho từng chunk.
3. Quét số slot ring trên IQ3_S (`STRATA_PREFILL_RING`) cùng hit rate expert cache; nếu hit rate dưới
   ~0.95 thì đây mới là nút thắt, không phải kernel.

Chỉ sau ba phép đo đó mới đặt lại con số prompt cho tài liệu này. **Không** cam kết 1,000 tok/s: với
đỉnh FP32 7.4 TFLOPS, con số đó nằm ngoài khả năng của card này ở mọi shape.

## 7. Những gì chưa kiểm chứng trong tài liệu này

- ~~Không có số đo IQ3_S trên bất kỳ gfx906 nào~~ — đã có: mục 4.1b đo kernel expert IQ3_S trên
  MI50 32 GB một card, ngày 18/10. Các số IQ3_S trong QUANTS.md vẫn là R9700; các số end-to-end
  cộng đồng vẫn là IQ2_XS.
- Một request chat 68-token prompt qua server đang chạy trên chính card này (`--spec 3`, `--kv int8`,
  `--resident-experts`, IQ3_S coder): **3.59–3.94 s cho 159 token ≈ 40–44 tok/s** (lần đầu 6.80 s do
  nguội cache). Đây là số của nhánh `Strata-v0.1.45`, không phải của nhánh này — chưa so sánh A/B.
- Config mục 3 suy ra từ hai báo cáo IQ2_XS, chưa chạy với IQ3_S.
- Mục 6b **đã** đo `moe_mmq_gfx906` và đường `dequant_f16` + `gemm_f16` trên MI50 một card ngày 19/10
  (bảng trong 6b), ở shape tổng hợp chứ chưa phải dòng GGUF thật qua `mmq_expert_bench`. **Chưa** đo
  end-to-end với dispatch mới (default tắt), và con số 50 MFLOP/token cho attention vẫn tính từ shape
  mặc định trong `geometry.hpp`, chưa đọc từ checkpoint thật.
- `udot4` và cờ inline LLVM: **vẫn chưa có số đo cô lập** (parity pass, nhưng chưa bật/tắt từng cái
  một trên cùng một run). Cờ `-mllvm -amdgpu-early-inline-all=true` có thật trong `flags.make` của
  `strata_kernels`; cảnh báo "argument unused" chỉ xuất hiện ở bước compile host.
- Mục "Not done" của `AMD_HIP.md` (dòng ~700) còn ghi: setup.py detection, Windows, images
  (`--vision`), MI60 và Radeon VII, **a single-card run**, tensor-split experiment.
