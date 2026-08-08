#pragma once
/// Multi-Head Attention — naive implementation (correctness baseline).
/// Phase 3+ will add FlashAttention-style tiled kernel.

#include "lightllm/tensor.h"

namespace lightllm {
namespace ops {

/// Scaled dot-product attention for a single batch of tokens.
///
/// q: [num_tokens, num_q_heads, head_dim]
/// k: [num_tokens, num_kv_heads, head_dim]
/// v: [num_tokens, num_kv_heads, head_dim]
///
/// Returns: [num_tokens, num_q_heads, head_dim]
///
/// Supports GQA: num_q_heads must be a multiple of num_kv_heads.
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v);

}  // namespace ops
}  // namespace lightllm
