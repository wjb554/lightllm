#pragma once
/// Element-wise CUDA kernels: add, mul, scale, silu, etc.
///
/// All kernels follow the pattern: one thread per element.

#include "lightllm/tensor.h"

namespace lightllm {
namespace ops {

// --- arithmetic ---
Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor scale(const Tensor& x, float factor);

// --- activation ---
Tensor silu(const Tensor& x);          // SiLU(x) = x * sigmoid(x)
void silu_inplace(Tensor& x);          // in-place SiLU

}  // namespace ops
}  // namespace lightllm
