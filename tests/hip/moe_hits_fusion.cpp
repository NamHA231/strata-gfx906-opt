// Compare fusion with the actual unfused GPU pipeline, including stale non-hit rows.
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}
template<class T> static T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc((void**)&d, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) ck(cudaMemcpy(d, h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
int main() {
    using namespace strata::kernels;
    std::mt19937 rng(906);
    std::uniform_real_distribution<float> value(-3, 3);
    cudaStream_t stream;
    ck(cudaStreamCreate(&stream));
    int failed = 0, cases = 0;
    for (int n : {1, 2, 3, 4, 8}) for (int width : {128, 2048, 2560, 4100})
    for (int mode = 0; mode < 4; ++mode) {
        std::vector<float> parts(n*10*width), hits(parts.size(), std::numeric_limits<float>::quiet_NaN());
        std::vector<float> weights(n*10), shared(n*width), gates(n);
        for (auto& x : parts) x = mode == 3 ? -0.0f : value(rng);
        for (auto& x : weights) x = mode == 3 ? 0.0f : value(rng);
        for (auto& x : shared) x = mode == 3 ? -0.0f : value(rng);
        for (auto& x : gates) x = value(rng)*30;
        std::vector<int32_t> dst;
        for (int r = 0; r < n*10; ++r) if (mode >= 2 || (mode == 1 && r%3 == 1)) {
            dst.push_back(r);
            for (int c = 0; c < width; ++c) hits[r*width+c] = mode == 3 ? -0.0f : value(rng);
        }
        std::shuffle(dst.begin(), dst.end(), rng);
        auto p = upload(parts), h = upload(hits), w = upload(weights), s = upload(shared), g = upload(gates);
        auto d = upload(dst), count = upload(std::vector<int32_t>{(int32_t)dst.size()});
        auto ref_parts = upload(parts), ref = upload(shared), got = upload(shared);
        moe_hit_add(ref_parts, h, d, count, n*10, width, stream);
        for (bool gated : {false, true}) {
            if (gated) native_moe_combine_multi_hits_gated(ref_parts, w, s, g, ref, width, 10, n, stream);
            else native_moe_combine_multi(ref_parts, w, s, ref, width, 10, n, stream);
            native_moe_combine_multi_hits_added(p, h, d, count, w, s, got, width, 10, n, stream, gated ? g : nullptr);
            ck(cudaStreamSynchronize(stream));
            std::vector<float> a(shared.size()), b(shared.size());
            ck(cudaMemcpy(a.data(), ref, a.size()*4, cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(b.data(), got, b.size()*4, cudaMemcpyDeviceToHost));
            if (std::memcmp(a.data(), b.data(), a.size()*4)) {
                ++failed;
                for (size_t i = 0; i < a.size(); ++i) if (std::memcmp(&a[i], &b[i], 4)) {
                    std::printf("FAIL T=%d width=%d mode=%d gated=%d index=%zu ref=%a got=%a\n", n, width, mode, gated, i, a[i], b[i]);
                    break;
                }
            }
            ++cases;
        }
        for (void* ptr : {(void*)p,(void*)h,(void*)w,(void*)s,(void*)g,(void*)d,(void*)count,(void*)ref_parts,(void*)ref,(void*)got}) ck(cudaFree(ptr));
    }
    ck(cudaStreamDestroy(stream));
    std::printf("moe_hits_fusion: %d cases, %d failures\n", cases, failed);
    return failed ? 1 : 0;
}
