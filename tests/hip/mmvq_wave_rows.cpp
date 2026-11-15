// gfx906: compare one versus two output rows per wave on the same Q8_0 weights.
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

static void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}

int main() {
    using namespace strata::kernels;
    std::mt19937 rng(906);
    cudaStream_t stream; ck(cudaStreamCreate(&stream));
    cudaEvent_t a, b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
    int failures = 0;
    for (auto [ni, no, t] : {std::tuple{2560, 2560, 1}, std::tuple{2560, 2560, 3},
                              std::tuple{2560, 640, 3}, std::tuple{640, 2560, 3}}) {
        std::vector<unsigned char> w((size_t) no * (ni / 32) * 34), x((size_t) t * (ni / 32) * 36);
        for (auto& v : w) v = (unsigned char) rng();
        for (auto& v : x) v = (unsigned char) rng();
        for (size_t i = 0; i < w.size(); i += 34) { w[i] = 0; w[i + 1] = 0x38; }
        for (size_t i = 0; i < x.size(); i += 36) { x[i] = 0; x[i+1] = 0x38; x[i+2] = 0; x[i+3] = 0x38; }
        void *dw, *dx; float* dy;
        ck(cudaMalloc(&dw, w.size())); ck(cudaMalloc(&dx, x.size()));
        ck(cudaMalloc((void**) &dy, (size_t) t * no * 4));
        ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dx, x.data(), x.size(), cudaMemcpyHostToDevice));
        std::vector<float> ref((size_t) t * no), got(ref.size());
        native_mmvq_set_gfx906_wave_rows(1);
        native_mmvq(8, dw, dx, dy, ni, no, t, stream); ck(cudaStreamSynchronize(stream));
        ck(cudaMemcpy(ref.data(), dy, ref.size()*4, cudaMemcpyDeviceToHost));
        native_mmvq_set_gfx906_wave_rows(2);
        native_mmvq(8, dw, dx, dy, ni, no, t, stream); ck(cudaStreamSynchronize(stream));
        ck(cudaMemcpy(got.data(), dy, got.size()*4, cudaMemcpyDeviceToHost));
        if (std::memcmp(ref.data(), got.data(), ref.size()*4)) {
            ++failures;
            std::printf("FAIL %d x %d T=%d\n", ni, no, t);
        }
        float ms[2] = {};
        for (int rows : {1, 2}) {
            native_mmvq_set_gfx906_wave_rows(rows);
            for (int k = 0; k < 20; ++k) native_mmvq(8, dw, dx, dy, ni, no, t, stream);
            ck(cudaStreamSynchronize(stream));
            ck(cudaEventRecord(a, stream));
            for (int k = 0; k < 200; ++k) native_mmvq(8, dw, dx, dy, ni, no, t, stream);
            ck(cudaEventRecord(b, stream)); ck(cudaEventSynchronize(b));
            ck(cudaEventElapsedTime(&ms[rows - 1], a, b));
        }
        std::printf("Q8_0 %d x %d T=%d: R1 %.2f us, R2 %.2f us, %s\n", ni, no, t,
                    ms[0] * 5, ms[1] * 5, failures ? "DIFF" : "bitwise equal");
        ck(cudaFree(dw)); ck(cudaFree(dx)); ck(cudaFree(dy));
    }
    native_mmvq_set_gfx906_wave_rows(1);
    ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b)); ck(cudaStreamDestroy(stream));
    return failures ? 1 : 0;
}
