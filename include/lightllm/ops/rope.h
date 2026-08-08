#pragma once
/// Rotary Position Embedding (RoPE) — Llama-style Neox rotation.

#include "lightllm/tensor.h"

namespace lightllm {
namespace ops {

/// Apply RoPE to query and optionally key.
/// q: [num_tokens, num_heads, head_dim]
/// k: [num_tokens, num_kv_heads, head_dim] or null
/// cos/sin: [num_tokens, rotary_dim / 2]
void rope(Tensor& q, Tensor* k, const Tensor& cos, const Tensor& sin);

/// Create cos/sin tensors on GPU for `num_tokens` positions.
/// Returns {cos, sin} each of shape [num_tokens, head_dim/2].
std::pair<Tensor,Tensor> make_rope_cos_sin(int num_tokens, int head_dim, float rope_theta);

}  // namespace ops
}  // namespace lightllm
