// src/prefill/moe_dot4_gfx906.cu - int8 v_dot4 expert GEMM for AMD MI50 (gfx906), measured 2026-10-21.
//
// IQ3_S weights are requantized IN VRAM to int8 + fp32 scale/32 with NO loss (IQ3_S grid magnitudes are
// integers 0..15, verified max=15 over the 512-entry ggml table): int8 = +/-grid, scale = d*(1+2*nibble).
// Measured 1,048,576/1,048,576 elements bit-exact vs ggml's iq3_s decoder (bench/dot-probe/n8_requant.hip).
// The GEMM is wave64-native: block 256, tile 64M x 64N, wave-tile 32x32 (lane micro-tile 4x4), k-tile 128
// in LDS with b128 quad swizzle, i32 accumulators flushed per 32-element scale block, prefetch of the next
// k-tile. bench/dot-probe/n6q.hip: vs hipBLAS HGEMM f16 on MI50 32 GB / ROCm 6.3.3, median of 20:
// gate/up T=40 1.44x, T=128 1.20x, T=512 1.02x; down T=512 1.12x (9.2-9.5 TF/s int8).
#include "strata/prefill/moe_dot4_gfx906.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <mutex>

namespace strata::prefill::gfx906 {
namespace {

// ---------------------------------------------------------------- IQ3_S -> int8 + fp32 scale/32 (bit-exact)
__device__ __forceinline__ void iq3_decode_block(const block_iq3_s* __restrict__ x, int8_t* __restrict__ qo,
                                                 float* __restrict__ so) {
    const float d = __half2float(x->d);
    #pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const int nib = (ib & 1) ? (x->scales[ib / 2] >> 4) : (x->scales[ib / 2] & 0xf);
        so[ib] = d * (1.0f + 2.0f * (float) nib);
        const uint8_t qh = x->qh[ib];
        const uint8_t* qs = x->qs + ib * 8;
        const uint8_t* sg = x->signs + ib * 4;
        int8_t* qo2 = qo + ib * 32;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t hi = ((uint32_t) qh << (8 - 2 * l)) & 256u;
            const uint32_t g1 = iq3s_grid[qs[2 * l] | hi], g2 = iq3s_grid[qs[2 * l + 1] | hi];
            const uint8_t m = sg[l];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t v1 = (g1 >> (8 * j)) & 0xff, v2 = (g2 >> (8 * j)) & 0xff;
                qo2[8 * l + j]     = (m & (1 << j))      ? -(int8_t) v1 : (int8_t) v1;
                qo2[8 * l + j + 4] = (m & (1 << (j + 4))) ? -(int8_t) v2 : (int8_t) v2;
            }
        }
    }
}

// ---------------------------------------------------------------- FP16 -> int8 + fp32 scale/32 (activations)
__global__ void iq3_to_q8_kernel(const block_iq3_s* __restrict__ src, int8_t* __restrict__ q,
                                 float* __restrict__ s, int nblk) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    iq3_decode_block(src + b, q + (size_t)b * 256, s + b * 8);
}

__global__ void quant_fp16_q8_kernel(const uint16_t* __restrict__ x, int8_t* __restrict__ q,
                                     float* __restrict__ s, int64_t nelem) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one thread per 32-elem block
    if (i * 32 >= nelem) return;
    const uint16_t* p = x + i * 32;
    float v[32], amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) { v[j] = __half2float(__ushort_as_half(p[j])); amax = fmaxf(amax, fabsf(v[j])); }
    const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / sc;
    s[i] = sc;
    int8_t* qo = q + i * 32;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        float r = v[j] * inv;
        qo[j] = (int8_t) lroundf(fmaxf(-127.0f, fminf(127.0f, r)));
    }
}

// ---------------------------------------------------------------- tiled dot4 GEMM
#define DOT4_SW(row, q) ((row) * 32 + (((q) ^ ((row) & 7)) << 2))

__launch_bounds__(256, 1) __global__ void gemm_dot4_kernel(
        const int8_t* __restrict__ Aq, const float* __restrict__ dA,
        const int8_t* __restrict__ Bq, const float* __restrict__ dB,
        float* __restrict__ C, int M, int N, int K, int ldy, float beta) {
    __shared__ int as[64 * 32], bs[64 * 32];
    __shared__ float das[64 * 4], dbs[64 * 4];
    const int n0 = blockIdx.x * 64, m0 = blockIdx.y * 64, t = threadIdx.x;
    const int row = t >> 2, cq = t & 3;
    const int wid = t >> 6, lane = t & 63;
    const int rb = (wid >> 1) * 32 + (lane & 7) * 4, cb = (wid & 1) * 32 + (lane >> 3) * 4;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    #pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    int4 va, vb, va4, vb4, va2, vb2, va6, vb6;
    const int gm = m0 + row;
    #define LD(p, off) ((gm < M) ? *(const int4*) ((p) + (off)) : make_int4(0, 0, 0, 0))
    va  = LD(Aq, (size_t) gm * K + cq * 16);        va4 = LD(Aq, (size_t) gm * K + 64 + cq * 16);
    vb  = *(const int4*) (Bq + (size_t) (n0 + row) * K + cq * 16);
    vb4 = *(const int4*) (Bq + (size_t) (n0 + row) * K + 64 + cq * 16);
    for (int k0 = 0; k0 < K; k0 += 128) {
        *(int4*) &as[DOT4_SW(row, cq)]     = va;  *(int4*) &bs[DOT4_SW(row, cq)]     = vb;
        *(int4*) &as[DOT4_SW(row, cq + 4)] = va4; *(int4*) &bs[DOT4_SW(row, cq + 4)] = vb4;
        {
            const int r = t >> 2, sb = t & 3;
            das[r * 4 + sb] = (m0 + r < M) ? dA[(size_t) (m0 + r) * (K / 32) + (k0 >> 5) + sb] : 0.0f;
            dbs[r * 4 + sb] = dB[(size_t) (n0 + r) * (K / 32) + (k0 >> 5) + sb];
        }
        __syncthreads();
        if (k0 + 128 < K) {
            va2 = LD(Aq, (size_t) gm * K + k0 + 128 + cq * 16); va6 = LD(Aq, (size_t) gm * K + k0 + 192 + cq * 16);
            vb2 = *(const int4*) (Bq + (size_t) (n0 + row) * K + k0 + 128 + cq * 16);
            vb6 = *(const int4*) (Bq + (size_t) (n0 + row) * K + k0 + 192 + cq * 16);
        }
        #pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            int ia[4][4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) ia[i][j] = 0;
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                int4 Aa[4], Bb[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) { const int r = rb + i; Aa[i] = *(int4*) &as[r * 32 + (((sb * 2 + q) ^ (r & 7)) << 2)]; }
                #pragma unroll
                for (int j = 0; j < 4; ++j) { const int c = cb + j; Bb[j] = *(int4*) &bs[c * 32 + (((sb * 2 + q) ^ (c & 7)) << 2)]; }
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    ia[i][j] = __builtin_amdgcn_sdot4(Aa[i].x, Bb[j].x, ia[i][j], false);
                    ia[i][j] = __builtin_amdgcn_sdot4(Aa[i].y, Bb[j].y, ia[i][j], false);
                    ia[i][j] = __builtin_amdgcn_sdot4(Aa[i].z, Bb[j].z, ia[i][j], false);
                    ia[i][j] = __builtin_amdgcn_sdot4(Aa[i].w, Bb[j].w, ia[i][j], false);
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                acc[i][j] += (float) ia[i][j] * das[(rb + i) * 4 + sb] * dbs[(cb + j) * 4 + sb];
        }
        __syncthreads();
        va = va2; vb = vb2; va4 = va6; vb4 = vb6;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gm2 = m0 + rb + i;
        if (gm2 < M) {
            float* c = C + (size_t) gm2 * ldy + n0 + cb;
            if (beta != 0.0f) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) c[j] = acc[i][j] + beta * c[j];
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) c[j] = acc[i][j];
            }
        }
    }
}

// ---------------------------------------------------------------- scratch + weight requant cache
struct Buf { void* p = nullptr; size_t cap = 0; };
Buf g_wq, g_ws, g_xq, g_xs;
std::mutex g_mu;

bool grow(Buf& b, size_t need) {
    if (need <= b.cap) return true;
    if (b.p) (void) hipFree(b.p);
    const size_t sz = std::max<size_t>(need * 2, 8 * 1024 * 1024);
    if (hipMalloc(&b.p, sz) != hipSuccess) { (void) hipGetLastError(); b.p = nullptr; b.cap = 0; return false; }
    b.cap = sz;
    return true;
}

} // namespace

bool moe_dot4_gfx906_supported(int ggml_type) { return ggml_type == GGML_TYPE_IQ3_S; }

bool moe_dot4_gfx906_run(const uint16_t* X, int ggml_type, const void* W_blocks,
                         float* Y, int64_t T, int64_t N, int64_t K,
                         int64_t ldy, float beta, void* stream) {
    if (!moe_dot4_gfx906_supported(ggml_type)) return false;
    if (T <= 0 || N <= 0 || K <= 0) return true;
    if (K % 128 != 0 || N % 64 != 0) return false;          // tiling contract; caller falls back
    if (ldy <= 0) ldy = N;
    cudaStream_t s = (cudaStream_t) stream;

    std::lock_guard<std::mutex> lk(g_mu);
    if (!grow(g_wq, (size_t) N * K) || !grow(g_ws, (size_t) N * K / 32 * 4) ||
        !grow(g_xq, (size_t) T * K) || !grow(g_xs, (size_t) T * K / 32 * 4)) return false;

    // Step 1: IQ3_S -> int8 + fp32 scale/32, EVERY call: expert-cache slots are recycled (a stable pointer can
    // carry different weights after an eviction), so a pointer-keyed cache could serve stale int8.  The pass is
    // 0.034 ms for a 1280x2560 expert (bench/dot-probe/n8_requant.hip), invisible next to the GEMM at engine T.
    const int nblk = (int) ((size_t) N * K / 256);
    iq3_to_q8_kernel<<<(nblk + 255) / 256, 256, 0, s>>>((const block_iq3_s*) W_blocks,
                                                        (int8_t*) g_wq.p, (float*) g_ws.p, nblk);

    // Step 2: FP16 activations -> int8 + fp32 scale/32.
    const int64_t nelem = T * K;
    quant_fp16_q8_kernel<<<(unsigned) ((nelem / 32 + 255) / 256), 256, 0, s>>>(X, (int8_t*) g_xq.p, (float*) g_xs.p, nelem);

    // Step 3: tiled dot4 GEMM.
    gemm_dot4_kernel<<<dim3((unsigned) (N / 64), (unsigned) ((T + 63) / 64)), 256, 0, s>>>(
        (const int8_t*) g_xq.p, (const float*) g_xs.p, (const int8_t*) g_wq.p, (const float*) g_ws.p,
        Y, (int) T, (int) N, (int) K, (int) ldy, beta);
    return true;
}

} // namespace strata::prefill::gfx906
