#pragma once
/// Matrix multiplication — cuBLAS, naive, and tiled shared-memory backends.

#include "lightllm/tensor.h"

namespace lightllm {
namespace ops {

enum class GemmBackend { CuBLAS, Naive, Tiled, Vectorized, DoubleBuf };

void set_gemm_backend(GemmBackend b);
GemmBackend get_gemm_backend();

/// C = A @ B  (all fp32, row-major)
Tensor gemm(const Tensor& a, const Tensor& b, bool transpose_b = false);

/// Mixed precision: A(fp16) @ B(fp16) → C(fp32)
/// Uses Tensor Cores for compute, accumulates in fp32.
Tensor gemm_f16f32(const Tensor& a, const Tensor& b, bool transpose_b = false);

}  // namespace ops
}  // namespace lightllm
