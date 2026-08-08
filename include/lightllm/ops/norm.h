#pragma once
/// RMSNorm — used by Llama as pre-attention and pre-MLP normalization.
///
///   rms = sqrt(mean(x^2) + eps)
///   y   = x / rms * weight

#include "lightllm/tensor.h"

namespace lightllm {
namespace ops {

/// Apply RMSNorm row-wise (each row is independently normalized).
/// x:      [num_tokens, hidden_size]
/// weight: [hidden_size]
/// eps:    small constant for numerical stability
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-6f);

}  // namespace ops
}  // namespace lightllm
