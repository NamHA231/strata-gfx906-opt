// N7 dot-probe part 1 (dot4 + fma32): phân xử xem ROCm 6.3.3 có phát v_dot4_i32_i8/v_dot4_u32_u8
// trên gfx906 hay hạ xuống v_macx_u8/v_madak. Bám pattern benchmark A (chained) + ILP4 của wiki-gfx906.
#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s @%d\n", hipGetErrorString(e), __LINE__); return 1; } } while (0)

// ---- (a) builtin udot4, accumulator phụ thuộc (chained) ----
extern "C" __global__ void k_dot4_builtin_chained(const int* __restrict__ a,
                                                  const int* __restrict__ b,
                                                  int* out, int iters) {
    int x = a[threadIdx.x], y = b[threadIdx.x], acc = 0;
    for (int i = 0; i < iters; ++i) acc = __builtin_amdgcn_udot4(x, y, acc, 0);
    if (acc == 0x7fffffff) out[blockIdx.x] = acc; // chống DCE
}
// ---- (a') builtin udot4 ILP-4 accumulator ----
extern "C" __global__ void k_dot4_builtin_ilp4(const int* __restrict__ a,
                                               const int* __restrict__ b,
                                               int* out, int iters) {
    int x = a[threadIdx.x], y = b[threadIdx.x];
    int acc0 = 0, acc1 = 1, acc2 = 2, acc3 = 3;
    for (int i = 0; i < iters; ++i) {
        acc0 = __builtin_amdgcn_udot4(x,     y, acc0, 0);
        acc1 = __builtin_amdgcn_udot4(x + 1, y, acc1, 0);
        acc2 = __builtin_amdgcn_udot4(x, y + 1, acc2, 0);
        acc3 = __builtin_amdgcn_udot4(x + 1, y + 1, acc3, 0);
    }
    if ((acc0 ^ acc1 ^ acc2 ^ acc3) == 0x7fffffff) out[blockIdx.x] = acc0;
}
// ---- (b) inline asm v_dot4_u32_u8 (ép ISA — compiler không được thay bằng v_macx) ----
extern "C" __global__ void k_dot4_asm_chained(const int* __restrict__ a,
                                              const int* __restrict__ b,
                                              int* out, int iters) {
    int x = a[threadIdx.x], y = b[threadIdx.x], acc = 0;
    for (int i = 0; i < iters; ++i)
        asm volatile("v_dot4_u32_u8 %0, %1, %2, %3" : "=v"(acc) : "v"(x), "v"(y), "v"(acc));
    if (acc == 0x7fffffff) out[blockIdx.x] = acc;
}
extern "C" __global__ void k_dot4_asm_ilp4(const int* __restrict__ a,
                                           const int* __restrict__ b,
                                           int* out, int iters) {
    int x = a[threadIdx.x], y = b[threadIdx.x];
    int acc0 = 0, acc1 = 1, acc2 = 2, acc3 = 3;
    for (int i = 0; i < iters; ++i) {
        asm volatile("v_dot4_u32_u8 %0, %1, %2, %3" : "=v"(acc0) : "v"(x),     "v"(y),     "v"(acc0));
        asm volatile("v_dot4_u32_u8 %0, %1, %2, %3" : "=v"(acc1) : "v"(x + 1), "v"(y),     "v"(acc1));
        asm volatile("v_dot4_u32_u8 %0, %1, %2, %3" : "=v"(acc2) : "v"(x),     "v"(y + 1), "v"(acc2));
        asm volatile("v_dot4_u32_u8 %0, %1, %2, %3" : "=v"(acc3) : "v"(x + 1), "v"(y + 1), "v"(acc3));
    }
    if ((acc0 ^ acc1 ^ acc2 ^ acc3) == 0x7fffffff) out[blockIdx.x] = acc0;
}
// ---- (d) FMA32 baseline (cùng hình thức chained / ILP4) ----
extern "C" __global__ void k_fma_chained(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         int* out, int iters) {
    float x = a[threadIdx.x], y = b[threadIdx.x], acc = 0.f;
    for (int i = 0; i < iters; ++i) acc = __fmaf_rn(acc, x, y);
    if (acc == 12345.678f) out[blockIdx.x] = 1;
}
extern "C" __global__ void k_fma_ilp4(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      int* out, int iters) {
    float x = a[threadIdx.x], y = b[threadIdx.x];
    float a0 = 0.f, a1 = 1.f, a2 = 2.f, a3 = 3.f;
    for (int i = 0; i < iters; ++i) {
        a0 = __fmaf_rn(a0, x, y); a1 = __fmaf_rn(a1, x + 1.f, y);
        a2 = __fmaf_rn(a2, x, y + 1.f); a3 = __fmaf_rn(a3, x + 1.f, y + 1.f);
    }
    if (a0 + a1 + a2 + a3 == 12345.678f) out[blockIdx.x] = 1;
}


#define BLK 256
#define GRID 2048
#define ITERS 65536

struct Case { const char* name; void (*launch)(int*,int*,int*,int); double flops_per_thread_iter; };

static void L4bc(int*a,int*b,int*o,int n){ k_dot4_builtin_chained<<<GRID,BLK>>>(a,b,o,n); }
static void L4bi(int*a,int*b,int*o,int n){ k_dot4_builtin_ilp4<<<GRID,BLK>>>(a,b,o,n); }
static void L4ac(int*a,int*b,int*o,int n){ k_dot4_asm_chained<<<GRID,BLK>>>(a,b,o,n); }
static void L4ai(int*a,int*b,int*o,int n){ k_dot4_asm_ilp4<<<GRID,BLK>>>(a,b,o,n); }
static void Lfc(int*a,int*b,int*o,int n){ k_fma_chained<<<GRID,BLK>>>((float*)a,(float*)b,o,n); }
static void Lfi(int*a,int*b,int*o,int n){ k_fma_ilp4<<<GRID,BLK>>>((float*)a,(float*)b,o,n); }

Case cases[] = {
    {"dot4_builtin_chained", L4bc, 8.0},
    {"dot4_builtin_ilp4",    L4bi, 32.0},
    {"dot4_asm_chained",     L4ac, 8.0},
    {"dot4_asm_ilp4",        L4ai, 32.0},
    {"fma32_chained",        Lfc,  2.0},
    {"fma32_ilp4",           Lfi,  8.0},
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
