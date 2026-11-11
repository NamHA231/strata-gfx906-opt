// src/prefill/moe_mmq_gfx906.cu - Native packed v_dot4_i32_i8 MMQ for AMD MI50 (gfx906).
#include "strata/prefill/moe_mmq_gfx906.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace strata::prefill::gfx906 {
namespace {

// ---------------------------------------------------------------- Device Helpers
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}

__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) {
    return ((const int*) x)[i32];
}

__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}

__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
#if defined(STRATA_HIP_GFX906)
    const uint32_t* v32 = reinterpret_cast<const uint32_t*>(table);
    const uint32_t q_even = (uint32_t) q4, q_odd = (uint32_t) q4 >> 4;
    const uint32_t el = __builtin_amdgcn_perm(v32[1], v32[0], q_even & 0x07070707u);
    const uint32_t ol = __builtin_amdgcn_perm(v32[1], v32[0], q_odd & 0x07070707u);
    const uint32_t eh = __builtin_amdgcn_perm(v32[3], v32[2], q_even & 0x07070707u);
    const uint32_t oh = __builtin_amdgcn_perm(v32[3], v32[2], q_odd & 0x07070707u);
    return make_int2((int) __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1)),
                     (int) __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1)));
#else
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
#endif
}

#define ggml_cuda_dp4a(a, b, c) STRATA_DP4A((a), (b), (c))

// ---------------------------------------------------------------- Dot Products
__device__ __forceinline__ float vec_dot_q2_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi0 = ggml_cuda_dp4a(grid0, u0, sumi0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi1 = ggml_cuda_dp4a(grid1, u1, sumi1);
    }
    const int ls = aux32 >> 27 | 1;
    const int sumi = (sumi0 + sumi1) * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F, ls1 = bq2->scales[iqs / 2] >> 4;
    int s00 = 0, s01 = 0, s10 = 0, s11 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            s00 = ggml_cuda_dp4a(grid_l, u0, s00);
            s01 = ggml_cuda_dp4a(grid_h, u1, s01);
        } else {
            s10 = ggml_cuda_dp4a(grid_l, u0, s10);
            s11 = ggml_cuda_dp4a(grid_h, u1, s11);
        }
    }
    const int sumi0 = s00 + s01, sumi1 = s10 + s11;
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int s00 = 0, s01 = 0, s10 = 0, s11 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            s00 = ggml_cuda_dp4a(grid_l, u0, s00);
            s01 = ggml_cuda_dp4a(grid_h, u1, s01);
        } else {
            s10 = ggml_cuda_dp4a(grid_l, u0, s10);
            s11 = ggml_cuda_dp4a(grid_h, u1, s11);
        }
    }
    const int sumi0 = s00 + s01, sumi1 = s10 + s11;
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
        sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
    }
    const int ls = aux32 >> 28;
    int sumi = sumi0 + sumi1;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                        iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
        sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
    }
    const int sumi = (sumi0 + sumi1) * (1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F));
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    const int aux_q4_0 = get_int_b2(bq4->qs, iqs + 0);
    const int aux_q4_1 = get_int_b2(bq4->qs, iqs + 1);
    const int2 v0 = get_int_from_table_16(aux_q4_0, kvalues_iq4nl);
    const int2 v1 = get_int_from_table_16(aux_q4_1, kvalues_iq4nl);
    int s0 = ggml_cuda_dp4a(v0.x, q8[0], 0);
    int s1 = ggml_cuda_dp4a(v0.y, q8[4], 0);
    s0 = ggml_cuda_dp4a(v1.x, q8[1], s0);
    s1 = ggml_cuda_dp4a(v1.y, q8[5], s1);
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * (s0 + s1);
}

__device__ __forceinline__ float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_q8_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q8_0* bq8_0 = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i)
        sumi = ggml_cuda_dp4a(get_int_b2(bq8_0->qs, iqs + i), get_int_b4(bq8_1->qs, iqs + i), sumi);
    const float d8_0 = __half2float(bq8_0->d), d8_1 = __low2float(bq8_1->ds);
    return d8_0 * d8_1 * ((float) sumi);
}

__device__ __forceinline__ int q4_0_centred(const int v) {
    const uint32_t b = (uint32_t) v ^ 0x08080808u;
    return (int) (b | ((b & 0x08080808u) * 0x1Eu));
}

__device__ __forceinline__ float vec_dot_q4_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q4_0* bq4 = (const block_q4_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = get_int_b2(bq4->qs, iqs + i);
        sumi = ggml_cuda_dp4a(q4_0_centred((v >> 0) & 0x0F0F0F0F), get_int_b4(bq8_1->qs, iqs + i), sumi);
        sumi = ggml_cuda_dp4a(q4_0_centred((v >> 4) & 0x0F0F0F0F), get_int_b4(bq8_1->qs, iqs + i + 16), sumi);
    }
    return __half2float(bq4->d) * __low2float(bq8_1->ds) * sumi;
}

__device__ __forceinline__ float vec_dot_q4_1_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q4_1* bq4_1 = (const block_q4_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = get_int_b4(bq4_1->qs, iqs + i);
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + 16);
        sumi = ggml_cuda_dp4a((v >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi = ggml_cuda_dp4a((v >> 4) & 0x0F0F0F0F, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const float2 dm4 = __half22float2(bq4_1->dm);
    const float d8 = __low2float(bq8_1->ds);
    return sumi * (dm4.x * d8) + sumu * (dm4.y * d8);
}

__device__ __forceinline__ float vec_dot_q5_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_0* bq5_0 = (const block_q5_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = get_int_b2(bq5_0->qs, iqs + i);
        const int vh = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + 16);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
    }
    const float d5 = __half2float(bq5_0->d);
    const float d8 = __low2float(bq8_1->ds);
    const float s8 = __high2float(bq8_1->ds);
    return d5 * (sumi * d8 - (16.0f * 2 / 16) * s8);
}

__device__ __forceinline__ float vec_dot_q5_1_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_1* bq5_1 = (const block_q5_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = get_int_b4(bq5_1->qs, iqs + i);
        const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + 16);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const float2 dm5 = __half22float2(bq5_1->dm);
    const float d8 = __low2float(bq8_1->ds);
    return sumi * (dm5.x * d8) + sumu * (dm5.y * d8);
}

// ---------------------------------------------------------------- Type Traits
template<int TY> struct Fmt;
template<> struct Fmt<GGML_TYPE_IQ3_S>   { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<GGML_TYPE_IQ4_NL>  { static constexpr int qk = 32,  ipb = 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<GGML_TYPE_IQ2_XXS> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<GGML_TYPE_IQ2_XS>  { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<GGML_TYPE_IQ2_S>   { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<GGML_TYPE_IQ3_XXS> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<GGML_TYPE_IQ4_XS>  { static constexpr int qk = 256, ipb = 8, step = 4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<GGML_TYPE_Q8_0>    { static constexpr int qk = 32,  ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q8_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<GGML_TYPE_Q4_0>    { static constexpr int qk = 32,  ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<GGML_TYPE_Q4_1>    { static constexpr int qk = 32,  ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<GGML_TYPE_Q5_0>    { static constexpr int qk = 32,  ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<GGML_TYPE_Q5_1>    { static constexpr int qk = 32,  ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<GGML_TYPE_Q2_0>    { static constexpr int qk = 64,  ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };

// ---------------------------------------------------------------- FP16 -> Q8_1 Quantization Kernel
__global__ void quantize_fp16_to_q8_1_kernel(const uint16_t* __restrict__ x,
                                             block_q8_1* __restrict__ y,
                                             int64_t total_elements) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) return;

    const float xi = __half2float(__ushort_as_half(x[i]));

    float amax = fabsf(xi);
    float sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }

    const float d = strata::kernels::q8_1_finite(amax / 127.0f);
    const int8_t q = strata::kernels::q8_1_quant(xi, d, amax);

    const int64_t ib = i / 32;
    const int iqs = (int) (i % 32);
    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = strata::kernels::q8_1_ds(d, sum);
    }
}

// ---------------------------------------------------------------- Tiled Wave64 MMQ Kernel
// Workgroup: dim3(64, 4) = 4 wavefronts of 64 lanes each.
// Wave w in [0, 3] handles Token t = blockIdx.x * 4 + w.
// Block processes NR=4 rows: n0 = blockIdx.y * 4 .. n0 + 3.
template<int TY, int NR = 4>
__global__ void __launch_bounds__(256) moe_mmq_gemm_gfx906_kernel(
        const block_q8_1* __restrict__ Xq,
        const uint8_t* __restrict__ W,
        size_t row_bytes,
        float* __restrict__ Y,
        int64_t T,
        int64_t N,
        int64_t K,
        int64_t ldy,
        float beta) {

    using F = Fmt<TY>;

    const int64_t t = (int64_t) blockIdx.x * 4 + threadIdx.y;
    if (t >= T) return;

    const int lane = threadIdx.x;
    const int64_t n0 = (int64_t) blockIdx.y * NR;

    const int nb = (int) (K / F::qk);
    const int n_calls = nb * F::ipb;
    const int64_t act_row_blocks = K / 32;
    const block_q8_1* x_row = Xq + t * act_row_blocks;

    float acc[NR];
#pragma unroll
    for (int j = 0; j < NR; ++j) acc[j] = 0.0f;

    for (int k = lane; k < n_calls; k += 64) {
        const int kbx = k / F::ipb;
        const int iqs = F::step * (k % F::ipb);
        const block_q8_1* xk = x_row + kbx * (F::qk / 32);

#pragma unroll
        for (int j = 0; j < NR; ++j) {
            if (n0 + j < N) {
                const uint8_t* wr = W + (size_t) (n0 + j) * row_bytes;
                acc[j] += F::dot(wr, xk, kbx, iqs);
            }
        }
    }

    // 64-lane tree reduction
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
            acc[j] += __shfl_xor(acc[j], o, 64);
        }
    }

    // Store result: lane 0 of each wave writes its token's NR outputs
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
            if (n0 + j < N) {
                float val = acc[j];
                if (beta != 0.0f) {
                    val += beta * Y[t * ldy + n0 + j];
                }
                Y[t * ldy + n0 + j] = val;
            }
        }
    }
}

// Global activation quantization buffer
static void* g_xq_buf = nullptr;
static size_t g_xq_cap = 0;
static std::mutex g_xq_mu;

} // namespace

void quantize_fp16_to_q8_1(const uint16_t* X, void* Y_q8_1, int64_t T, int64_t K, void* stream) {
    const int64_t total = T * K;
    if (total <= 0) return;
    const unsigned threads = 256;
    const unsigned blocks = (unsigned) ((total + threads - 1) / threads);
    quantize_fp16_to_q8_1_kernel<<<blocks, threads, 0, (cudaStream_t) stream>>>(
        X, (block_q8_1*) Y_q8_1, total);
}

bool moe_mmq_gfx906_supported(int ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q2_0:
            return true;
        default:
            return false;
    }
}

bool moe_mmq_gfx906_run(const uint16_t* X, int ggml_type, const void* W_blocks,
                        float* Y, int64_t T, int64_t N, int64_t K,
                        int64_t ldy, float beta, void* stream) {
    if (!moe_mmq_gfx906_supported(ggml_type)) return false;
    if (T <= 0 || N <= 0 || K <= 0) return true;
    if (K % 32 != 0) return false;
    if (ldy <= 0) ldy = N;

    const size_t rb = strata::kernels::iq_row_bytes(ggml_type, K);
    if (rb == 0) return false;

    // Allocate / grow activation quantization buffer
    const size_t act_bytes = (size_t) T * (size_t) (K / 32) * sizeof(block_q8_1);
    void* xq_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_xq_mu);
        if (act_bytes > g_xq_cap) {
            if (g_xq_buf) (void) hipFree(g_xq_buf);
            const size_t alloc_sz = std::max<size_t>(act_bytes * 2, 16 * 1024 * 1024);
            if (hipMalloc(&g_xq_buf, alloc_sz) != hipSuccess) {
                (void) hipGetLastError();
                g_xq_buf = nullptr;
                g_xq_cap = 0;
                return false;
            }
            g_xq_cap = alloc_sz;
        }
        xq_ptr = g_xq_buf;
    }

    cudaStream_t s = (cudaStream_t) stream;

    // Step 1: Quantize FP16 X -> Q8_1 Xq
    quantize_fp16_to_q8_1(X, xq_ptr, T, K, s);

    // Step 2: Launch Tiled MMQ GEMM
    constexpr int NR = 4;
    const dim3 block(64, 4);
    const dim3 grid((unsigned) ((T + 3) / 4), (unsigned) ((N + NR - 1) / NR));

    const auto* W = (const uint8_t*) W_blocks;
    const auto* Xq = (const block_q8_1*) xq_ptr;

    switch (ggml_type) {
#define LAUNCH_CASE(TYPE) \
        case TYPE: \
            moe_mmq_gemm_gfx906_kernel<TYPE, NR><<<grid, block, 0, s>>>( \
                Xq, W, rb, Y, T, N, K, ldy, beta); \
            break;

        LAUNCH_CASE(GGML_TYPE_IQ3_S)
        LAUNCH_CASE(GGML_TYPE_IQ4_NL)
        LAUNCH_CASE(GGML_TYPE_IQ2_XXS)
        LAUNCH_CASE(GGML_TYPE_IQ2_XS)
        LAUNCH_CASE(GGML_TYPE_IQ2_S)
        LAUNCH_CASE(GGML_TYPE_IQ3_XXS)
        LAUNCH_CASE(GGML_TYPE_IQ4_XS)
        LAUNCH_CASE(GGML_TYPE_Q8_0)
        LAUNCH_CASE(GGML_TYPE_Q4_0)
        LAUNCH_CASE(GGML_TYPE_Q4_1)
        LAUNCH_CASE(GGML_TYPE_Q5_0)
        LAUNCH_CASE(GGML_TYPE_Q5_1)
        LAUNCH_CASE(GGML_TYPE_Q2_0)
#undef LAUNCH_CASE
        default:
            return false;
    }

    return true;
}

} // namespace strata::prefill::gfx906

