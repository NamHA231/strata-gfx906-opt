// Engine-shape bench for the dot4 path (moe_dot4_gfx906) - same cases/format as mmq_expert_bench.
#include "ggml.h"
#include "strata/prefill/moe_dot4_gfx906.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

#define CK(x) do{ auto _e=(x); if(_e){fprintf(stderr,"%s @%d\n",hipGetErrorString(_e),__LINE__); exit(1);} }while(0)
namespace dot4 = strata::prefill::gfx906;

int main(int argc, char** argv) {
    std::vector<int> Es = {16};          // experts per launch like the engine
    std::vector<int> Rs = {12, 160, 626};
    if (argc > 1) { Rs.clear(); for (int i = 1; i < argc; ++i) Rs.push_back(atoi(argv[i])); }
    struct Case { const char* n; int64_t rows, cols; };  // engine: N=rows, K=cols
    const Case cases[] = {{"iq3_s gate/up 1280x2560", 1280, 2560}, {"iq3_s down 2560x640", 2560, 640}};
    std::mt19937 rng(7);
    std::printf("dot4_expert_bench: %d experts a launch, medians of 25\n", Es[0]);
    for (int R : Rs) for (const Case& c : cases) {
        const int T = Es[0] * R;
        const size_t nblk = (size_t) Es[0] * c.rows * c.cols / 256;
        std::vector<uint8_t> w(nblk * 110);
        for (auto& b : w) b = (uint8_t) (rng() & 255);
        for (size_t i = 0; i < nblk; ++i) { uint16_t d = 0x2000 + (uint16_t) (rng() & 255); memcpy(w.data() + i * 110, &d, 2); }
        std::vector<uint16_t> x((size_t) T * c.cols);
        std::uniform_real_distribution<float> nd(-2.f, 2.f);
        for (auto& h : x) { float v = nd(rng); uint16_t hh; // f16 via hip conversion host: quick manual
            // use simple f32->f16
            float f = v; uint32_t u; memcpy(&u, &f, 4);
            uint32_t sign = (u >> 16) & 0x8000; int e = ((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffff;
            uint16_t h16; if (e <= 0) h16 = sign; else if (e >= 31) h16 = sign | 0x7c00; else h16 = sign | (e << 10) | (m >> 13);
            hh = h16; h = hh; }
        uint8_t* dw; uint16_t* dx; float* dd;
        CK(hipMalloc(&dw, w.size())); CK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
        CK(hipMalloc(&dx, x.size() * 2)); CK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
        CK(hipMalloc(&dd, (size_t) T * c.rows * 4));
        auto call = [&]{ dot4::moe_dot4_gfx906_run(dx, GGML_TYPE_IQ3_S, dw, dd, T, c.rows, c.cols, c.rows, 0.f, nullptr); };
        call(); CK(hipDeviceSynchronize());
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
        std::vector<float> ts;
        for (int i = 0; i < 28; ++i) { hipEventRecord(e0); call(); hipEventRecord(e1);
            CK(hipEventSynchronize(e1)); float ms; hipEventElapsedTime(&ms, e0, e1); if (i >= 3) ts.push_back(ms); }
        std::sort(ts.begin(), ts.end());
        const float ms = ts[ts.size() / 2];
        printf("  R %4d  %-24s %8.3f ms  %7.1f TOPS(eqv)\n", R, c.n, ms, 2.0 * T * (double) c.rows * c.cols / (ms * 1e9));
        hipFree(dw); hipFree(dx); hipFree(dd); hipEventDestroy(e0); hipEventDestroy(e1);
    }
    return 0;
}
