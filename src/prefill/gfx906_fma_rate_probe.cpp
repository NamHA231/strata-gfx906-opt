// src/prefill/gfx906_fma_rate_probe.cpp - Issue-rate probe for the three math paths a prompt GEMM
// could use on AMD Instinct MI50 (gfx906): scalar v_fma_f32, packed v_pk_fma_f16, and the udot4
// byte-dot chain. It answers one question with a stopwatch, not a spec sheet: is there any way on
// this card to execute more multiply-adds per clock than 64 FP32 FMAs per lane-clock?
//
// Everything here is a dependent-free accumulation loop over a value read from memory, so the
// compiler cannot fold it and cannot hoist it out. The kernel touches no HBM in the loop; it is a
// pure issue-rate measurement. Compare the printed TFLOP/s against the card's FP32 peak
// (60 CU x 64 lanes x 2 FLOP/FMA x boost clock) to get the fraction of peak each path reaches.
//
// Build (see docs/AMD_HIP.md:641-670): ninja -C build-906 gfx906_fma_rate_probe
// Run:  ./build-906/gfx906_fma_rate_probe
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdio>
#include <cstdlib>

#define CHECK_HIP(call) do { \
    hipError_t err_ = (call); \
    if (err_ != hipSuccess) { \
        std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err_)); \
        std::exit(1); \
    } \
} while (0)

// Eight independent accumulators, so the loop is limited by issue rate and not by FMA latency.
// `sink` keeps the operands loop-invariant-but-unknown; the kernel reads them from device memory.
__global__ void f32_probe(float* out, int iters) {
    const float a = out[threadIdx.x & 31];
    const float b = out[(threadIdx.x & 31) + 32];
    float a0 = a, a1 = a + 1.f, a2 = a + 2.f, a3 = a + 3.f;
    float a4 = a + 4.f, a5 = a + 5.f, a6 = a + 6.f, a7 = a + 7.f;
    for (int i = 0; i < iters; ++i) {
        a0 = __fmaf_rn(a0, b, a); a1 = __fmaf_rn(a1, b, a);
        a2 = __fmaf_rn(a2, b, a); a3 = __fmaf_rn(a3, b, a);
        a4 = __fmaf_rn(a4, b, a); a5 = __fmaf_rn(a5, b, a);
        a6 = __fmaf_rn(a6, b, a); a7 = __fmaf_rn(a7, b, a);
    }
    float s = (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)));
    if (s == 12345.678f) out[threadIdx.x] = s;  // never true; keeps the chain alive
}

__global__ void f16x2_probe(float* out, int iters) {
    const __half hb = __float2half(0.9999f);
    const __half2 b = __half2half2(hb);
    const __half ha = __float2half(out[threadIdx.x & 31]);
    __half2 a = __half2half2(ha);
    __half2 h0 = a, h1 = a, h2 = a, h3 = a, h4 = a, h5 = a, h6 = a, h7 = a;
    for (int i = 0; i < iters; ++i) {
        h0 = __hfma2(h0, b, a); h1 = __hfma2(h1, b, a);
        h2 = __hfma2(h2, b, a); h3 = __hfma2(h3, b, a);
        h4 = __hfma2(h4, b, a); h5 = __hfma2(h5, b, a);
        h6 = __hfma2(h6, b, a); h7 = __hfma2(h7, b, a);
    }
    float2 f0 = __half22float2(__hadd2(h0, h1));
    float2 f1 = __half22float2(__hadd2(h2, h3));
    float2 f2 = __half22float2(__hadd2(h4, h5));
    float2 f3 = __half22float2(__hadd2(h6, h7));
    float s = (f0.x + f0.y) + (f1.x + f1.y) + (f2.x + f2.y) + (f3.x + f3.y);
    if (s == 12345.678f) out[threadIdx.x] = s;
}

// The byte-dot path. On gfx906 udot4 has no V_DOT hardware instruction and lowers to a
// v_macx_u8 / v_madak_u8 chain, so the question is whether that chain issues faster than one FMA.
__global__ void udot4_probe(float* out, int iters) {
    const uint32_t a = (uint32_t) out[threadIdx.x & 31];
    const uint32_t b = a ^ 0x01020304u;
    int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int i = 0; i < iters; ++i) {
        a0 = __builtin_amdgcn_udot4(a, b, a0, false);
        a1 = __builtin_amdgcn_udot4(b, a, a1, false);
        a2 = __builtin_amdgcn_udot4(a, a, a2, false);
        a3 = __builtin_amdgcn_udot4(b, b, a3, false);
    }
    int32_t s = (a0 + a1) + (a2 + a3);
    if (s == 1234567) out[threadIdx.x] = (float) s;
}

struct Result { double ms; double ops_per_sec; int insns_per_iter; };

template <typename K>
Result run(K kernel, int blocks, int threads, int iters, double flops_per_insn, const char* name) {
    float* out = nullptr;
    CHECK_HIP(hipMalloc(&out, 64 * sizeof(float)));
    CHECK_HIP(hipMemset(out, 0x3f, 64 * sizeof(float)));  // small positive floats, no zeros
    hipEvent_t e0, e1;
    CHECK_HIP(hipEventCreate(&e0));
    CHECK_HIP(hipEventCreate(&e1));
    kernel<<<blocks, threads>>>(out, iters);  // warm the clocks and the icache
    CHECK_HIP(hipGetLastError());
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        CHECK_HIP(hipEventRecord(e0));
        kernel<<<blocks, threads>>>(out, iters);
        CHECK_HIP(hipEventRecord(e1));
        CHECK_HIP(hipEventSynchronize(e1));
        float ms = 0.f;
        CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
        if (ms < best) best = ms;
    }
    CHECK_HIP(hipGetLastError());
    const double lanes = (double) blocks * threads;
    const double insns = lanes * (double) iters * 8.0;   // eight accumulators per iteration
    Result r{best, insns * flops_per_insn / (best * 1e-3), 8};
    std::printf("  %-14s  %10.3f ms   %8.2f GFLOP/s-equivalent   (%d accumulators/thread)\n",
                name, best, r.ops_per_sec / 1e9, r.insns_per_iter);
    CHECK_HIP(hipEventDestroy(e0));
    CHECK_HIP(hipEventDestroy(e1));
    CHECK_HIP(hipFree(out));
    return r;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    CHECK_HIP(hipSetDevice(0));
    hipDeviceProp_t p;
    CHECK_HIP(hipGetDeviceProperties(&p, 0));
    const double peak_f32 = (double) p.multiProcessorCount * 64.0 * 2.0 * ((double) p.clockRate * 1e-6);
    std::printf("=====================================================================================\n");
    std::printf("  gfx906 math issue-rate probe (%s)\n", p.name);
    std::printf("  %d CU, wave size %d, clock %.0f MHz\n", p.multiProcessorCount, p.warpSize,
                (double) p.clockRate * 1e-3);
    std::printf("  FP32 FMA peak, 64 lanes/CU: %.2f TFLOP/s\n", peak_f32 / 1e3);
    std::printf("=====================================================================================\n\n");

    const int threads = 256;
    const int blocks = p.multiProcessorCount * 16;   // deep occupancy, four waves per SIMD-pair x4
    const int iters = 200000;

    std::printf("  One FMA per slot for f32; two multiply-adds per slot for f16x2; four byte\n"
                "  products per slot for udot4. Divide by the peak above to get % of peak.\n\n");

    Result f32 = run(f32_probe, blocks, threads, iters, 2.0, "f32 fma");
    Result f16 = run(f16x2_probe, blocks, threads, iters, 4.0, "f16x2 hfma2");
    Result u4  = run(udot4_probe, blocks, threads, iters, 8.0, "udot4 byte-dot");

    std::printf("\n  Rate ratios against scalar FP32 (issue slots, not FLOP/s):\n");
    const double f32_rate = ((double) blocks * threads * iters * 8.0) / (f32.ms * 1e-3);
    const double f16_rate = ((double) blocks * threads * iters * 8.0) / (f16.ms * 1e-3);
    const double u4_rate  = ((double) blocks * threads * iters * 4.0) / (u4.ms * 1e-3);
    std::printf("    v_pk_fma_f16 : %.3f x a scalar FMA slot  -> %s\n", f16_rate / f32_rate,
                f16_rate / f32_rate > 1.5 ? "PACKED (2 multiply-adds per slot)"
                                          : "same rate as FP32: packed buys nothing");
    std::printf("    udot4        : %.3f x a scalar FMA slot  -> %s\n", u4_rate / f32_rate,
                u4_rate / f32_rate > 1.5 ? "faster than a scalar FMA per byte-product"
                                         : "same rate as FP32: no V_DOT, as expected on gfx906");
    std::printf("\n  Decision this probe settles: if v_pk_fma_f16 is PACKED, the FP32 ceiling of\n"
                "  %.2f TFLOP/s is not the prompt ceiling; a packed-FP16 GEMM can reach near 2x.\n",
                peak_f32 / 1e3);
    return 0;
}
