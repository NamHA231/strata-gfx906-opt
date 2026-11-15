# GFX906 / MI50 Decode — phân tích bottleneck & plan bứt phá (2026-11-12)

Tài liệu này tổng hợp từ: đo live hôm nay (rocprofiler trace trên chính binary `build-906-pr`,
kernel stats `bench/results/2026-11-12-gfx906-expert-tiling/`), `GFX906_DECODE_SPEED_DIAGNOSIS_0211.md`,
`GFX906_COMPLETE_OPTIMIZATION_REPORT.md`, `GFX906_MI50_PREFILL_RESEARCH.md`, và probe ISA
inline-asm chạy trên máy này hôm 12/11. Mọi con số dưới đây là **đo**, không suy đoán; chỗ nào ước
lượng thì ghi rõ [ước].

---

## 0. Hiện trạng đo được (nền để mọi gate so vào đây)

- Engine 0.1.57 (nhánh opt, build `build-906-pr`), decode live **39.2–47.3 tok/s** (metrics 8080).
- Cửa sổ verify: **T ≤ 4** (MTP spec=3, lookup đang tắt), trung bình **~2.1–2.8 token verified/cửa sổ**,
  verify wall **≈ 41–46 ms/cửa sổ**.
- Cấu hình live: `--spec 3`, MTP `mtp/rt` (đúng MTP-đầu-của-chính-model), `draft_vocab.bin` **đã bật**
  (draft head đã là subset-vocab), CRA bật (head full-vocab **chỉ 1 lần/cửa sổ**), lookup **0**,
  `--mtp-q4` chưa bật, window auto = 16 (HIP), EXP_MODE=8 (nibble experts).

## 1. Bốn sự thật đo được quyết định hướng đi

**(1) Decode bị GPU-thời-gian-bind, không phải host/launch-bind.**
Trace queue decode chính hôm nay: busy **94.1%**; host chờ GPU (GPU-reach wait) 34–39 ms/cửa sổ;
đợi doorbell giữa các pha (`wait_flag`) trung bình **5.3 µs × 74/cửa sổ ≈ 0.4 ms** — handshake
host↔GPU và kernel-launch gap (burst density gaps <1%) **không phải** bottleneck. Mọi kế hoạch
"mega-kernel để giết launch overhead" tự bác bằng số liệu này.

**(2) ~55–60% thời gian GPU decode là grouped-expert kernels, và chúng chạy ở 111–150 GB/s
trong khi 746 GB/s đã được chứng minh là đạt được trên chính card này.**
Phân bổ từ kernel stats hôm nay (39 cửa sổ, quy về /cửa sổ):

| thành phần | ms/cửa sổ | ghi chú |
|---|---|---|
| expert IQ grouped (mmvq_wave 54µs×3471, gu_amd 88µs×494, gu_fused 70µs×754, down_lds 41µs×1248, gr_down_staged 34.5µs×1348, gr_up_fast 28.9µs×1376) | **≈ 11.8** | bytes/call ⇒ **111–150 GB/s** hiệu dụng (diag 02/11 tính; khớp trace hôm nay) |
| head row (mmvf_multi 34.5µs×1464 — full vocab, CRA dedup chuẩn) | ≈ 1.3 | ~900 GB/s ✓ không còn gì để ăn |
| expert swap copy16_group 78.6µs×827 | ≈ 1.7 | traffic CPU→VRAM, có thể overlap (mục P2) |
| fillBuffer memset 318µs×250 | ≈ 2.0 | zero-buffer churn, sửa được (P2) |
| attention GDN + kv copy + norm 72.8µs×274 + 15.8µs×1730 + … | ≈ 1.4 | đã tối ưu (gating 11/2026) |
| PLE + misc | ≈ 1–2 | đã overlapped 50/50 (26/10) |

Tensile HSS/BSS ≈ 1.2 s trong profile là **prefill**, không liên quan decode.
Chẩn đoán cũ (02/11): expert kernel BW 111–300 GB/s; microbench Q8_0 cùng card đạt **746 GB/s**
⇒ **khe hở 5–6× là do cấu trúc LDS/latency trong kernel**, không phải HBM, không phải ISA-trần.

**(3) Card này KHÔNG có tensor core. Đơn vị "tensor" của nó là dot4/dot8.**
Probe hôm nay (clang 18 ROCm 6.3.3, inline asm + ISA dump) khớp docs `GFX906_MI50_PREFILL_RESEARCH.md`
(đo 21/10, 3 fork vLLM crash ILLEGAL_INSTRUCTION với `v_mfma*`):
- `v_mfma_*` / WMMA: **không tồn tại trên gfx906** (MFMA chỉ từ gfx908). Engine hiện zero-MFMA — đúng.
- **Có thật**: `v_dot4_i32_i8`, `v_dot4_u32_u8`, `v_dot8_i32_i4`, `v_dot2_f32_f16`
  (builtin `__builtin_amdgcn_{s,u}dot4/sdot8`, engine **đã dùng** trong `moe_dot4_gfx906.cu`
  (IQ3_S, shipped) và `moe_mmq_gfx906.cu` (MMQ v_dot4_i32_i8, prototype chạy đúng 21/10)).
- Đo đỉnh trong docs: dot4-ILP4 ≈ 22 TMAC/s (3.3× FMA32), **dot8-ILP4 ≈ 86 TOPS chạy thật**
  (đỉnh công thức dot8 = 106 TOPS — chính là con số "INT8 TOPS" marketing của MI50).
⇒ Mọi kế hoạch "thay MFMA cho 906" chết. Trần int-GEMM của 906 = dot4 (8-bit) / **dot8 (4-bit)**.

**(4) Mật độ lệnh KHÔNG phải tường chắn của expert kernel hôm nay — latency pipeline mới là.**
Toán kiểm chứng: expert kernels đọc ~12 ms × (111–150 GB/s) ⇒ demand MAC/instruction thấp hơn
nhiều so với dot4-peak (22 TMAC/s). Nghĩa là kernel **hiện chưa ăn hết VALU lẫn BW**; nó mắc ở
chuỗi LDS-load latency + wave occupancy thấp (mode8 "4-entry LDS stage" là dạng tuần tự).
Flip implication: khi đã pipeline tới ~500–700 GB/s, dot4-peak còn ~8× headroom VALU — và
`v_dot8_i32_i4` (8 MAC 4-bit/lệnh, **gấp 2×** dot4 trên chính weights nibble hiện tại, **không cần
requant, cùng số học**) là thứ mua thêm headroom + cắt một nửa số lệnh unpack trong chuỗi latency.

**Bài học hôm nay (bắt buộc thành protocol):** experiment wave-2-rows (`gfx906-mmvq-r2`): kernel
**+30–50% standalone nhưng −0.76% end-to-end, không reproduce**. Win cô lập ≠ win hệ thống.
⇒ Mọi phase dưới đây chỉ được đóng票 bằng: **window-hash A/B + đổi thứ tự chạy + ≥2 runs/arm**
(văn hoá bit-perfect sẵn có của engine).

**Còn một ẩn số phần cứng:** card đang chạy **cap 115 W** (đặt từ phiên trước, chưa verify live hôm
nay). Khi kernel còn latency-bound, clock ∝ thời gian gần tuyến tính ⇒ nới cap có thể là +5–15%
"miễn phí" — P0 đo trước khi tin.

---

## 2. Luận đề bứt phá (một câu)

> **Mang máy "GEMM-tiling Vega20" đã được chứng minh ở prefill (dot4/dot8 + LDS swizzle + prefetch
> + full-wave) vào các grouped-expert kernel của DECODE — đưa expert GEMV từ 111–150 GB/s về gần
> BW-bound (~500–700 GB/s) — đồng thời đào sâu cửa sổ verify (lookup draft trên nền MTP) để khấu
> hao toàn bộ chi phí cố định/cửa sổ, và nới cap điện nếu P0 chứng minh.**

Cộng hưởng hai chiều: cửa sổ sâu (T 8–16) làm kernel grouped dedup weights nhiều hơn (mỗi byte
weight nuôi nhiều token hơn ⇒ hiệu dụng BW/token tăng thêm), và kernel nhanh hơn làm chi phí
cố định/cửa sổ rẻ đi ⇒ đáng đào sâu thêm.

## 3. Plan 5 phase (thứ tự cứng, gate tường minh)

### P0 — Ground truth (0.5–1 ngày, zero code-sửa)
1. Decode-only per-kernel **ms/cửa sổ** trên binary live (tách prefill ra; dùng bench harness
   decode-only + rocprofiler; bảng mục 1 là tiên lượng, cần số chính xác).
2. Microbench floor: chạy `gfx906_mmq_bench` (đã tồn tại) với đúng shape expert decode
   (1024×1024, groups 4–16 experts, T=3..16, input nibble-cache layout) ⇒ **BW đích có thật** cho
   kernel grouped (không dùng con số 746 của layout khác).
3. Power: đọc cap hiện tại + clock/poder telemetry trong lúc decode bench; thử **cap 150 W**
   (`rocm-smi --setpoweroverdrive`, sudo) 15 phút, A/B decode. Rollback nếu nhiệt/PSU lỗi.
4. Ghi `bench/results/2026-11-12-p0-groundtruth/` (README + csv), chốt 3 số: `expert_ms_per_win`,
   `bw_target_gb_s`, `gain_from_clock`.
**Gate đi tiếp:** expert_ms_per_win ≥ 8 (dự kiến 11.8 ✓) → P3 đáng đầu tư. < 5 → chuyển trọng tâm
vào P2/P4.

### P1 — Config-only wins (1–2 ngày, chủ yếu thời gian bench)
Matrix trên binary y nguyên, mỗi arm ≥2 runs + order-swap + window-hash:
- `STRATA_LOOKUP_MAX_T=8` (mixed MTP+lookup lấp cửa sổ 8; engine đã hỗ trợ đầy đủ, đang TẮT).
  Traffic Hermes (turns template hoá) là kịch bản best-case của n-gram lookup; đo cả prose thường.
- `--mtp-q4` (draft proj/head 4-bit: draft compute 2–4 ms/cửa sổ giảm ~½; theo dõi acceptance).
- cap 150 W (từ P0).
- Kéo `--spec-min-p` 0.5→0.35 thử riêng trên agent-traffic (draft sai tốn slot, không tốn GPU).
Kỳ vọng **+8–14% tổng**; config revert dễ (git + env), log đầy đủ vào `bench/results/.../config-matrix/`.

### P2 — Dọn đuôi cố định/cửa sổ (2–4 ngày, code nhỏ, ~7–11 ms/cửa sổ đang nằm đây)
- **memset churn 2 ms**: fuse chỉ zero các hàng được ghi (pattern `copy_or_zero_from_mapped` đã
  có — mở rộng cho `parts_`/`hit_out` thay vì memset 318 µs × 250).
- **swap copy 1.7 ms**: overlap CPU→VRAM copy trên stream phụ (queue 4 đã tồn tại) trong lúc draft
  chạy, thay vì serial ở đầu cửa sổ.
- **quyết định working-tree hiện tại** (mmvq wave-2-rows + fuse-hits đang dirty): fold lever
  group-size vào P3 hay revert — theo số P0, không để lửng.
Gate: −2.0…−3.5 ms/cửa sổ trên P0-profile; E2E qua hash-protocol; kỳ vọng **+3–6%**.

### P3 — BỨT PHÁ: `native_expert_pipe_gfx906` — grouped decode kernel pipelined (2–4 tuần)
Kernel mới (file riêng, gate `STRATA_EXPERT_PIPE=0|1`, default off), giữ nguyên khế combine/`parts_`:
- K-tile 128, **LDS double-buffer** (GCN5: global→reg trong khi tile trước đang MAC, reg→LDS;
  pattern đã chạy trong `moe_mmq_gfx906.cu`), LDS **b128 swizzle zero-conflict**, prefetch.
- **Full-wave**: gộp 8–16 experts/CTA (group size 4→8–16) để mỗi byte weight nuôi ≥4 token khi
  T≥4 (chỉ hiệu lực khi P1 đã đào T — thứ tự P1→P3 là cố ý).
- Activation đã là **Q8_1 s8** (đầu vào sẵn của MMVQ) — stage LDS một lần/cửa sổ.
- Inner-loop MAC: **variant A = v_dot4_i32_i8 trên nibble→s8 (bit-parity với hôm nay ⇒ acceptance
  bit-perfect, zero rủi ro accuracy)** — mặc định. **Variant B = v_dot8_i32_i4 với act requant i4**
  (2× density, cắt nửa số VALU trong chuỗi latency) — chỉ ship nếu microbench + 300-token live
  probe cho acceptance không giảm >1đ; calibration bảng ghi vào `expert-profile.bin`.
- Scale flush theo block-32 (chu kỳ Q8_1) bằng FMA, 2 bước MFMA-less… 2 bước dot/flush.
Landing gates: (i) microbench ≥ 1.8× kernel cũ trên shape layer-8 (tests/hip harness có sẵn),
(ii) acceptance không đổi trên live probe, (iii) **E2E ≥ +8% hoặc dừng-redesign** (bài học r2).
**Mục tiêu: expert 11.8 → 4–6 ms/cửa sổ ⇒ 45→38 ms ⇒ +18–25% decode** (stretch 3 ms ⇒ +30%).
Rollback = env off. Không requant ở variant A (dữ liệu nibble hiện dùng trực tiếp).

### P4 — Nhân hệ số: cửa sổ sâu (luồng Tuning, kẹp giữa các phase)
Sau P1+P3: distribution T 6–9 verified/cửa sổ ⇒ head 1.3 + memset 2 + attn 1.4 + swap 1.7 ms
 amortised gấp đôi trên mỗi token; retune `lookup_max_t` 8→12, window cap 16 (auto đã cho),
spec_min_p theo acceptance thực đo trên traffic Hermes (metrics 8080 tok/s + window-hash + swap p99).
Optional P4b: 2-level speculation (MTP3 + 5–6 lookup slots = window 8) — engine đã hỗ trợ mixed.
Kỳ vọng **+5–8% cộng thêm**, và nó là điều kiện để P3 variant B (dot8) đáng bật.

### Hoãn lại / KHÔNG làm (kèm số liệu bác bỏ, khỏi quay lại vòng cũ)
- **Mega-kernel/persistent CTA bây giờ**: launch gap + doorbell waits đo được <6% wall (mục 1.1).
  Revisit nếu sau P3 cái đuôi per-layer (post-combine gap trung bình ~205 µs × ~50 layer ≈ 3 ms)
  vượt 15% cửa sổ — lúc đó làm mega theo lớp, experts resident.
- **fp16 weights** (26/10: 16 bpw, BW+VRAM sụp), **MFMA/TC** (ISA không tồn tại — đo lại hôm nay),
  **host-handshake rewrite** (doorbell 5.3 µs), **DFlash2 thay MTP** (chậm hơn, 17.9 vs 22–26 tok/s),
  **PLE→GPU** (gather 50×), **act f16→int8** (đã xong: Q8_1), **int8-expert re-pack toàn bộ**
  (2.6× bytes, slot VRAM-cache giảm ⇒ hit-rate rơi — chỉ đáng nếu dot8-variant thắng lớn).

## 4. Dự phóng cộng dồn (lấy gate làm thật, không lấy kỳ vọng)

| phase | gate đo được | đóng góp kỳ vọng |
|---|---|---|
| P0 clock/cap | decode bench A/B ≥ +3% | +5–15% |
| P1 lookup/mtp-q4 | tok/s + acceptance không giảm | +8–14% |
| P2 đuôi | −2 ms/cửa sổ | +3–6% |
| P3 pipe kernel | micro ≥1.8×, E2E ≥+8% | +18–25% |
| P4 window sâu | verified tokens/cửa sổ ≥5.5 TB trung bình | +5–8% |

Toàn bộ: **39–47 → ~60–75 tok/s**. Trường hợp chỉ P0/P1/P2 đậu gate: **50–55**. Số xấu nhất có thật
(P3 trượt gate): ~+10–20% từ config/dẹp đuôi — đã là đáng làm.

## 5. Sổ rủi ro
- R1 act-i4 (dot8) mất accuracy → variant A bit-parity là fallback, gate acceptance 300-token probe.
- R2 lookup sâu làm acceptance prose giảm → ship theo workload (config, không code).
- R3 150 W: nhiệt/PSU → P0 đo 15 phút + telemetry, revert một dòng.
- R4 LDS double-buffer 64 KB/CTA ⇒ occupancy CTA giảm → tune buffer 32 KB / 2-stage.
- R5 (đã thành protocol): win cô lập không extrapolate — mọi phase đóng bằng hash-protocol.

## 6. Việc đầu tiên khi được bật đèn
P0-1: capture decode-only profile trên binary live (workload 39k-token đã có sẵn script),parse ra
bảng ms/cửa sổ theo nhóm (experts/head/memset/swap/attn/PLE), rồi P0-2 microbench floor song song.
Không sửa một dòng code nào trước khi bảng P0 nằm trong `bench/results/2026-11-12-p0-groundtruth/README.md`.
