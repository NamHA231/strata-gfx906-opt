#pragma once
#include <cstdint>

namespace strata::prefill::gfx906 {

bool moe_dot4_gfx906_supported(int ggml_type);

// int8 x int8 v_dot4 tiled GEMM path (IQ3_S weights requantized bit-exactly to int8 + fp32 scale/32 in VRAM).
// Returns false when the shape/type is not handled (caller falls back to the FP16 path).
bool moe_dot4_gfx906_run(const uint16_t* X, int ggml_type, const void* W_blocks,
                         float* Y, int64_t T, int64_t N, int64_t K,
                         int64_t ldy, float beta, void* stream);

} // namespace strata::prefill::gfx906
