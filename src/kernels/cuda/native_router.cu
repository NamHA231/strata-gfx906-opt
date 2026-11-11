// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_router.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}
__launch_bounds__(256, 1)
__global__ void route(const float* __restrict__ logits, int32_t* __restrict__ ids,
                      float* __restrict__ weights) {
    // Preserve the pinned 32x8 block geometry; only row zero is active here.
    // blockIdx.x = the token (a multi-token launch; 0 for the single one)
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
__launch_bounds__(256, 1)
__global__ void route_multi(const float* __restrict__ logits, int32_t* __restrict__ ids,
                            float* __restrict__ weights, int n_tok) {
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    logits += (size_t) tk * 512; ids += (size_t) tk * 10; weights += (size_t) tk * 10;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncwarp();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
// The same contract for other (experts, top-k) shapes - Qwen3.6-35B-A3B's 256 / 8 - one warp per token, n_tok tokens
// (8 per block).  The 512 / 10 path keeps its own kernels above (the Flash-Next bits), this template is only used
// for the other shapes.
template <int NE, int K>
__launch_bounds__(256, 1)
__global__ void route_topk(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                           int n_tok) {
    static_assert(NE % 32 == 0 && K <= 32, "experts in whole warps, at most 32 picks");
    constexpr int V = NE / 32;
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    logits += (size_t) tk * NE; ids += (size_t) tk * K; weights += (size_t) tk * K;
    const int lane = threadIdx.x;
    float values[V];
#pragma unroll
    for (int i = 0; i < V; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < V; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < K; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < V; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < K) weights[lane] = selected * inverse_selected_sum;
}
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    route<<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    route_multi<<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
bool native_router_topk_supported(int n_expert, int k) { return (n_expert == 512 && k == 10) || (n_expert == 256 && k == 8); }
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream) {
    if (n_expert == 512 && k == 10) {   // Flash-Next: the pinned kernels above, unchanged
        if (n_tok == 1) native_router_top10(logits, ids, weights, stream);
        else native_router_top10_multi(logits, ids, weights, n_tok, stream);
        return;
    }
    if (!stream || n_tok < 1 || !native_router_topk_supported(n_expert, k) ||
        !valid(logits, (size_t) n_tok * n_expert * 4) || !valid(ids, (size_t) n_tok * k * 4) ||
        !valid(weights, (size_t) n_tok * k * 4))
        throw std::invalid_argument("native_router_topk: a stream, a supported (experts, k) and aligned buffers");
    route_topk<256, 8><<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// STRATA_ROUTE_RESIDENT (EXPERIMENTAL, opt-in, changes the output): residency-biased routing for 512 x 10.
// After the top-10: for the ranks lo..hi a non-resident expert is replaced by the best resident expert not already
// picked when its logit is within `margin` of the replaced one; the weights are then the softmax of the selected
// logits (the router's renormalisation).  Tokens without a swap keep the router's exact bits.
// stats (device, 4 x uint64): [0] tail entries seen, [1] swaps, [2] non-resident entries before, [3] after.
//
// One warp per token.  This used to be one THREAD per token scanning all 512 experts (4 ranks x 512 x 12 scalar
// ops on up to 6 threads): 45 us per layer, 2.2 ms of a 19.4 ms window, which is exactly what the option saved in
// PCIe experts (~2.4 ms) - the swap cost as much as the misses it removed.  The ranks are still visited in order
// (a swap changes the set already picked), but the scan is coalesced (lane j reads f = j, j + 32, ...) and reduced
// within the warp.  The bits are unchanged: a strictly greater logit wins, the smallest expert index breaks ties,
// and the softmax is the same expression.
__global__ void route_resident_k(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                 const int32_t* __restrict__ res, int n_tok, float margin, int lo, int hi,
                                 unsigned long long* __restrict__ stats) {
    const int t = blockIdx.x;
    if (t >= n_tok) return;
    const int lane = (int) threadIdx.x;
    const float* l = logits + (size_t) t * 512;
    int32_t* id = ids + (size_t) t * 10;
    int32_t my[10];
    for (int r = 0; r < 10; ++r) my[r] = id[r];
    int swaps = 0, tail = 0, before = 0;
    if (lane == 0)
        for (int r = 0; r < 10; ++r) before += res[my[r]] < 0;
    for (int r = lo; r <= hi && r < 10; ++r) {
        const int e = my[r];
        if (res[e] >= 0) continue;                      // already resident: uniform across the warp
        if (lane == 0) ++tail;
        float bv = -INFINITY;
        int bi = -1;
        for (int f = lane; f < 512; f += 32) {
            if (res[f] < 0) continue;
            bool used = false;
            for (int q = 0; q < 10; ++q) used |= my[q] == f;
            if (used) continue;
            if (l[f] > bv) { bv = l[f]; bi = f; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
        }
        if (lane == 0 && bi >= 0 && l[e] - bv <= margin) { my[r] = bi; ++swaps; }
        my[r] = __shfl_sync(0xffffffffu, my[r], 0);     // all lanes keep the same picked set
    }
    int after = 0;
    if (lane == 0)
        for (int r = 0; r < 10; ++r) after += res[my[r]] < 0;
    const int sw = __shfl_sync(0xffffffffu, swaps, 0);
    if (sw) {
        float m = -INFINITY;
        for (int r = 0; r < 10; ++r) m = fmaxf(m, l[my[r]]);
        float ex[10], sum = 0.0f;
        for (int r = 0; r < 10; ++r) { ex[r] = expf(l[my[r]] - m); sum += ex[r]; }
        for (int r = 0; r < 10; ++r) weights[(size_t) t * 10 + r] = ex[r] / sum;
    }
    if (lane == 0) {
        for (int r = 0; r < 10; ++r) id[r] = my[r];
        if (stats) {
            atomicAdd(stats + 0, (unsigned long long) tail);
            atomicAdd(stats + 1, (unsigned long long) sw);
            atomicAdd(stats + 2, (unsigned long long) before);
            atomicAdd(stats + 3, (unsigned long long) after);
        }
    }
}
// Co-Routed Asymmetric Zero-Stall Verify (CRA-ZSV):
// Single-block kernel for decode verify windows (n_tok <= 8):
// - On AMD gfx906 (64-lane wavefront), launched with dim3(64, 8) so each token t = threadIdx.y occupies its own
//   full 64-lane wavefront with zero intra-wavefront SIMT divergence between target (t == 0) and draft (t > 0),
//   and scans 512 experts in just 8 coalesced steps per lane (512 / 64 = 8).
// - Phase 1 (t == 0, Target / Prefix Token):
//   Preserves top ranks 0..lo-1 (default 0..4) untouched; replaces non-resident experts in ranks lo..hi (5..9) with
//   the best resident expert in VRAM within `margin` (for ranks lo..6) or `draft_margin` (for tail ranks 7..9 that
//   STRATA_ROUTE_TAIL_SKIP=7 would otherwise zero out), followed by exact softmax renormalisation.
//   Publishes token 0's final resident/selected expert set to `__shared__ int32_t s_t0[10]`.
// - Phase 2 (t > 0, Speculative Draft Tokens):
//   1. Co-Routed Speculative Coalescing (CRSC): for tail ranks r >= co_lo (default 5), if `my[r]` is not already in
//      token 0's picked set `s_t0[0..9]`, coalesces `my[r]` onto the highest-logit resident expert in `s_t0[0..9]`
//      not yet picked by token `t` if within `co_margin` logits. This merges distinct GPU expert groups across the
//      verify window without extra HBM2 weight reads.
//   2. Asymmetric Draft Residency: for ranks r >= draft_lo (default 1), if `my[r]` is still non-resident in VRAM,
//      replaces it with the best resident expert in VRAM (giving a `+co_bonus` logit preference to experts already
//      in `s_t0[0..9]`) within `draft_margin` (or `r1_margin` at rank 1), eliminating speculative CPU DDR4 misses
//      and 30 KB PCIe doorbell activation writes while keeping target verify distribution exact.
__global__ void route_resident_co_k(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                    const int32_t* __restrict__ res, int n_tok, float margin, int lo, int hi,
                                    float draft_margin, float r1_margin, int draft_lo,
                                    float co_margin, int co_lo, float co_bonus,
                                    unsigned long long* __restrict__ stats) {
#if defined(__HIP_PLATFORM_AMD__)
    constexpr int kW = 64;
#else
    constexpr int kW = 32;
#endif
    __shared__ int32_t s_t0[10];
    const int lane = (int) threadIdx.x;
    const int t = (int) threadIdx.y;
    const bool active = (t < n_tok);

    const float* l = active ? (logits + (size_t) t * 512) : nullptr;
    int32_t* id = active ? (ids + (size_t) t * 10) : nullptr;
    int32_t my[10];
    if (active) {
        #pragma unroll
        for (int r = 0; r < 10; ++r) my[r] = id[r];
    }

    int swaps = 0, tail = 0, before = 0;
    if (active && lane == 0) {
        for (int r = 0; r < 10; ++r) before += res[my[r]] < 0;
    }

    // ---- Phase 1: Target token (t == 0) residency routing (preserves top-3 r=0..2 exact) ----
    if (active && t == 0) {
        const int t0_lo = lo < 3 ? 3 : lo;
        for (int r = t0_lo; r <= hi && r < 10; ++r) {
            const int e = my[r];
            if (res[e] >= 0) continue;
            if (lane == 0) ++tail;
            // Smooth rank-progressive logit margin curve for target token (t == 0):
            // r=0..2: untouched (exact top-3 dominant experts)
            // r=3: 0.65f, r=4: 1.30f, r=5..9: draft_margin (>=3.50f)
            const float eff_margin = (r == 3) ? 0.65f
                                   : (r == 4) ? fmaxf(margin * 2.6f, 1.30f)
                                              : fmaxf(draft_margin, 3.50f);
            float bv = -INFINITY;
            int bi = -1;
            for (int f = lane; f < 512; f += kW) {
                if (res[f] < 0) continue;
                bool used = false;
                #pragma unroll
                for (int q = 0; q < 10; ++q) used |= (my[q] == f);
                if (used) continue;
                const float v = l[f];
                if (v > bv) { bv = v; bi = f; }
            }
            #pragma unroll
            for (int off = kW / 2; off > 0; off >>= 1) {
#if defined(__HIP_PLATFORM_AMD__)
                const float ov = __shfl_down_sync(0xffffffffu, bv, off, kW);
                const int oi = __shfl_down_sync(0xffffffffu, bi, off, kW);
#else
                const float ov = __shfl_down_sync(0xffffffffu, bv, off);
                const int oi = __shfl_down_sync(0xffffffffu, bi, off);
#endif
                if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
            }
            if (lane == 0 && bi >= 0 && l[e] - bv <= eff_margin) { my[r] = bi; ++swaps; }
#if defined(__HIP_PLATFORM_AMD__)
            my[r] = __shfl_sync(0xffffffffu, my[r], 0, kW);
#else
            my[r] = __shfl_sync(0xffffffffu, my[r], 0);
#endif
        }
        if (lane < 10) s_t0[lane] = my[lane];
    }

    __syncthreads();

    // ---- Phase 2: Speculative draft tokens (t > 0) Co-Routed Coalescing & Zero-Stall Residency ----
    // Preserves top-1 rank (r = 0) 100% exact on every draft row.
    if (active && t > 0) {
        // Step 2a: Co-Routed Speculative Coalescing (CRSC) on ranks r >= co_lo (default 2)
        if (co_margin > 0.0f) {
            for (int r = co_lo; r <= hi && r < 10; ++r) {
                const int e = my[r];
                bool in_t0 = false;
                #pragma unroll
                for (int q = 0; q < 10; ++q) in_t0 |= (s_t0[q] == e);
                if (in_t0 && res[e] >= 0) continue;
                if (lane == 0) {
                    float bv = -INFINITY;
                    int bi = -1;
                    #pragma unroll
                    for (int q = 0; q < 10; ++q) {
                        const int cand = s_t0[q];
                        if (cand < 0 || res[cand] < 0) continue;
                        bool used = false;
                        #pragma unroll
                        for (int u = 0; u < 10; ++u) used |= (my[u] == cand);
                        if (used) continue;
                        const float v = l[cand];
                        if (v > bv || (v == bv && (bi < 0 || cand < bi))) { bv = v; bi = cand; }
                    }
                    const float rank_m = (r <= 1) ? r1_margin
                                       : (r == 2) ? 1.50f
                                                  : fmaxf(draft_margin, 3.50f);
                    const float allow = (res[e] < 0) ? fmaxf(co_margin, rank_m) : co_margin;
                    if (bi >= 0 && l[e] - bv <= allow) {
                        if (res[e] < 0) ++tail;
                        my[r] = bi;
                        ++swaps;
                    }
                }
#if defined(__HIP_PLATFORM_AMD__)
                my[r] = __shfl_sync(0xffffffffu, my[r], 0, kW);
#else
                my[r] = __shfl_sync(0xffffffffu, my[r], 0);
#endif
            }
        }

        // Step 2b: Zero-Stall Draft Residency on remaining non-resident ranks r >= draft_lo (default 1)
        const int d_lo = draft_margin > 0.0f ? draft_lo : lo;
        for (int r = d_lo; r <= hi && r < 10; ++r) {
            const int e = my[r];
            if (res[e] >= 0) continue;
            if (lane == 0) ++tail;
            const float eff_margin = (r <= 1) ? r1_margin
                                   : (r == 2) ? 1.50f
                                              : fmaxf(draft_margin, 3.50f);
            float bv = -INFINITY;
            int bi = -1;
            for (int f = lane; f < 512; f += kW) {
                if (res[f] < 0) continue;
                bool used = false;
                #pragma unroll
                for (int q = 0; q < 10; ++q) used |= (my[q] == f);
                if (used) continue;
                bool in_t0 = false;
                #pragma unroll
                for (int q = 0; q < 10; ++q) in_t0 |= (s_t0[q] == f);
                const float score = l[f] + (in_t0 ? co_bonus : 0.0f);
                if (score > bv) { bv = score; bi = f; }
            }
            #pragma unroll
            for (int off = kW / 2; off > 0; off >>= 1) {
#if defined(__HIP_PLATFORM_AMD__)
                const float ov = __shfl_down_sync(0xffffffffu, bv, off, kW);
                const int oi = __shfl_down_sync(0xffffffffu, bi, off, kW);
#else
                const float ov = __shfl_down_sync(0xffffffffu, bv, off);
                const int oi = __shfl_down_sync(0xffffffffu, bi, off);
#endif
                if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
            }
            if (lane == 0 && bi >= 0 && l[e] - l[bi] <= eff_margin) { my[r] = bi; ++swaps; }
#if defined(__HIP_PLATFORM_AMD__)
            my[r] = __shfl_sync(0xffffffffu, my[r], 0, kW);
#else
            my[r] = __shfl_sync(0xffffffffu, my[r], 0);
#endif
        }
    }

    if (!active) return;

    int after = 0;
    if (lane == 0) {
        for (int r = 0; r < 10; ++r) after += res[my[r]] < 0;
    }
#if defined(__HIP_PLATFORM_AMD__)
    const int sw = __shfl_sync(0xffffffffu, swaps, 0, kW);
#else
    const int sw = __shfl_sync(0xffffffffu, swaps, 0);
#endif
    if (sw) {
        float m = -INFINITY;
        for (int r = 0; r < 10; ++r) m = fmaxf(m, l[my[r]]);
        float ex[10], sum = 0.0f;
        for (int r = 0; r < 10; ++r) { ex[r] = expf(l[my[r]] - m); sum += ex[r]; }
        for (int r = 0; r < 10; ++r) weights[(size_t) t * 10 + r] = ex[r] / sum;
    }
    if (lane == 0) {
        for (int r = 0; r < 10; ++r) id[r] = my[r];
        if (stats) {
            atomicAdd(stats + 0, (unsigned long long) tail);
            atomicAdd(stats + 1, (unsigned long long) sw);
            atomicAdd(stats + 2, (unsigned long long) before);
            atomicAdd(stats + 3, (unsigned long long) after);
        }
    }
}

void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok, float margin,
                           int rank_lo, int rank_hi, unsigned long long* stats, void* stream) {
    if (!stream || n_tok < 1 || !res_layer) throw std::invalid_argument("native_route_resident: bad arguments");
    static const bool co_on = [] {
        const char* v = std::getenv("STRATA_ROUTE_CO_RESIDENT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    static const float env_draft_margin = [] {
        const char* v = std::getenv("STRATA_DRAFT_ROUTE_RESIDENT");
        return v != nullptr ? (float) std::atof(v) : -1.0f;
    }();
    static const float env_r1_margin = [] {
        const char* v = std::getenv("STRATA_DRAFT_R1_MARGIN");
        return v != nullptr ? (float) std::atof(v) : 0.80f;
    }();
    static const int env_draft_lo = [] {
        const char* v = std::getenv("STRATA_DRAFT_RESIDENT_LO");
        return v != nullptr ? std::clamp(std::atoi(v), 0, 9) : 1;
    }();
    static const float env_co_margin = [] {
        const char* v = std::getenv("STRATA_CO_ROUTE_MARGIN");
        return v != nullptr ? (float) std::atof(v) : 0.30f;
    }();
    static const int env_co_lo = [] {
        const char* v = std::getenv("STRATA_CO_ROUTE_LO");
        return v != nullptr ? std::clamp(std::atoi(v), 1, 9) : 2;
    }();
    static const float env_co_bonus = [] {
        const char* v = std::getenv("STRATA_CO_ROUTE_BONUS");
        return v != nullptr ? (float) std::atof(v) : 0.25f;
    }();

    if (co_on && n_tok <= 8) {
#if defined(__HIP_PLATFORM_AMD__)
        const dim3 blk(64, 8);
#else
        const dim3 blk(32, 8);
#endif
        const float d_margin = env_draft_margin >= 0.0f ? env_draft_margin : fmaxf(margin * 6.0f, 3.5f);
        route_resident_co_k<<<1, blk, 0, static_cast<cudaStream_t>(stream)>>>(
            logits, ids, weights, res_layer, n_tok, margin, rank_lo, rank_hi,
            d_margin, env_r1_margin, env_draft_lo, env_co_margin, env_co_lo, env_co_bonus, stats);
    } else {
        route_resident_k<<<(unsigned) n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(
            logits, ids, weights, res_layer, n_tok, margin, rank_lo, rank_hi, stats);
    }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
