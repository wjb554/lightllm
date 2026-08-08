#pragma once
/// Reduction operators — reduce along last dimension.

#include "lightllm/tensor.h"

namespace lightllm {
namespace ops {

// Reduce along last dim: [*, D] → [*]
Tensor reduce_sum(const Tensor& x);
Tensor reduce_mean(const Tensor& x);
Tensor reduce_max(const Tensor& x);
Tensor reduce_min(const Tensor& x);

// TopK along last dim, returns (values, indices)
std::pair<Tensor, Tensor> topk(const Tensor& x, int k);

}  // namespace ops
}  // namespace lightllm
