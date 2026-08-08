#pragma once
/// Embedding lookup — gather rows from a weight matrix.
/// Equivalent to: output[i] = weight[token_ids[i]]

#include "lightllm/tensor.h"

namespace lightllm {
namespace ops {

/// Lookup embedding vectors from a [vocab, dim] weight matrix.
/// token_ids: [num_tokens] int32 on GPU
/// weight:    [vocab_size, hidden_size]
/// Returns:   [num_tokens, hidden_size]
Tensor embedding(const Tensor& token_ids, const Tensor& weight);

}  // namespace ops
}  // namespace lightllm
