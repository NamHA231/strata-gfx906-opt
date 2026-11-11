// N7 dot-probe part 2 (dot8): v_dot8_i32_i4 có tồn tại trên gfx906 không — assembler là trọng tài.
#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s @%d\n", hipGetErrorString(e), __LINE__); return 1; } } while (0)

#ifdef USE_BUILTIN
// ---- builtin sdot8/udot8 (nếu LLVM 18 gfx906 không có dot8, file này KHÔNG biên dịch được) ----
extern "C" __global__ void k_dot8_builtin_chained(const int* __restrict__ a,
                                                  const int* __restrict__ b,
                                                  int* out, int iters) {
    int x = a[threadIdx.x], y = b[threadIdx.x], acc = 0;
    for (int i = 0; i < iters; ++i) acc = __builtin_amdgcn_sdot8(x, y, acc, 0);
    if (acc == 0x7fffffff) out[blockIdx.x] = acc;
}
extern "C" __global__ void k_dot8_builtin_ilp4(const int* __restrict__ a,
                                               const int* __restrict__ b,
                                               int* out, int iters) {
    int x = a[threadIdx.x], y = b[threadIdx.x];
    int acc0 = 0, acc1 = 1, acc2 = 2, acc3 = 3;
    for (int i = 0; i < iters; ++i) {
        acc0 = __builtin_amdgcn_sdot8(x,     y,     acc0, 0);
        acc1 = __builtin_amdgcn_sdot8(x + 8, y,     acc1, 0);
        acc2 = __builtin_amdgcn_sdot8(x,     y + 8, acc2, 0);
        acc3 = __builtin_amdgcn_sdot8(x + 8, y + 8, acc3, 0);
    }
    if ((acc0 ^ acc1 ^ acc2 ^ acc3) == 0x7fffffff) out[blockIdx.x] = acc0;
}
#endif

#ifdef USE_ASM
// ---- inline asm: trọng tài tối thượng — assembler gfx906 accept hay reject? ----
extern "C" __global__ void k_dot8_asm_chained(const int* __restrict__ a,
                                              const int* __restrict__ b,
                                              int* out, int iters) {
    int x = a[threadIdx.x], y = b[threadIdx.x], acc = 0;
    for (int i = 0; i < iters; ++i)
        asm volatile("v_dot8_i32_i4 %0, %1, %2, %3" : "=v"(acc) : "v"(x), "v"(y), "v"(acc));
    if (acc == 0x7fffffff) out[blockIdx.x] = acc;
}
extern "C" __global__ void k_dot8_asm_ilp4(const int* __restrict__ a,
                                           const int* __restrict__ b,
                                           int* out, int iters) {
    int x = a[threadIdx.x], y = b[threadIdx.x];
    int acc0 = 0, acc1 = 1, acc2 = 2, acc3 = 3;
    for (int i = 0; i < iters; ++i) {
        asm volatile("v_dot8_i32_i4 %0, %1, %2, %3" : "=v"(acc0) : "v"(x),     "v"(y),     "v"(acc0));
        asm volatile("v_dot8_i32_i4 %0, %1, %2, %3" : "=v"(acc1) : "v"(x + 8), "v"(y),     "v"(acc1));
        asm volatile("v_dot8_i32_i4 %0, %1, %2, %3" : "=v"(acc2) : "v"(x),     "v"(y + 8), "v"(acc2));
        asm volatile("v_dot8_i32_i4 %0, %1, %2, %3" : "=v"(acc3) : "v"(x + 8), "v"(y + 8), "v"(acc3));
    }
    if ((acc0 ^ acc1 ^ acc2 ^ acc3) == 0x7fffffff) out[blockIdx.x] = acc0;
}
#endif


#define BLK 256
#define GRID 2048
#define ITERS 65536

struct Case { const char* name; void (*launch)(int*,int*,int*,int); double flops_per_thread_iter; };
Case cases[] = {
#ifdef USE_BUILTIN
    {"dot8_builtin_chained", [](int*a,int*b,int*o,int n){ k_dot8_builtin_chained<<<GRID,BLK>>>(a,b,o,n); }, 16.0},
    {"dot8_builtin_ilp4",    [](int*a,int*b,int*o,int n){ k_dot8_builtin_ilp4<<<GRID,BLK>>>(a,b,o,n); },    64.0},
#endif
#ifdef USE_ASM
    {"dot8_asm_chained",     [](int*a,int*b,int*o,int n){ k_dot8_asm_chained<<<GRID,BLK>>>(a,b,o,n); },     16.0},
    {"dot8_asm_ilp4",        [](int*a,int*b,int*o,int n){ k_dot8_asm_ilp4<<<GRID,BLK>>>(a,b,o,n); },        64.0},
#endif
};

int main() {
    int *da, *db, *dout;
    CHECK(hipMalloc(&da, BLK * sizeof(int)));
    CHECK(hipMalloc(&db, BLK * sizeof(int)));
    CHECK(hipMalloc(&dout, GRID * sizeof(int)));
    int ha[BLK], hb[BLK];
    for (int i = 0; i < BLK; ++i) { ha[i] = 0x01020304 + i; hb[i] = 0x04030201 - i; }
    CHECK(hipMemcpy(da, ha, sizeof(ha), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, hb, sizeof(hb), hipMemcpyHostToDevice));

    hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
    double thread_iters = (double)GRID * BLK * ITERS;
    printf("%-22s %10s %14s\n", "kernel", "ms", "TFLOP/s(FLOP)");
    for (auto& c : cases) {
        c.launch(da, db, dout, 1000); CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
        float best = 1e9f;
        for (int rep = 0; rep < 3; ++rep) {
            hipEventRecord(t0);
            c.launch(da, db, dout, ITERS);
            hipEventRecord(t1);
            CHECK(hipEventSynchronize(t1));
            float ms = 0; hipEventElapsedTime(&ms, t0, t1);
            if (ms < best) best = ms;
        }
        double tflops = thread_iters * c.flops_per_thread_iter / (best * 1e-3) / 1e12;
        printf("%-22s %10.2f %14.2f\n", c.name, best, tflops);
    }
    return 0;
}
