#pragma once
#include <cstdint>
#include <cstddef>

namespace strata::prefill::gfx906 {

bool moe_mmq_gfx906_supported(int ggml_type);

// Run native MMQ GEMM on gfx906 using packed v_dot4_i32_i8 instructions
bool moe_mmq_gfx906_run(const uint16_t* X, int ggml_type, const void* W_blocks,
                        float* Y, int64_t T, int64_t N, int64_t K,
                        int64_t ldy, float beta, void* stream);

// Direct FP16 to block_q8_1 quantization without FP32 scratch
void quantize_fp16_to_q8_1(const uint16_t* X, void* Y_q8_1, int64_t T, int64_t K, void* stream);

} // namespace strata::prefill::gfx906
