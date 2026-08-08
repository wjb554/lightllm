#pragma once
/// A single Transformer block (Qwen2/Llama architecture):
///   RMSNorm → Q/K/V projection (+bias) → RoPE → Attention → O projection → residual
///   RMSNorm → gate/up projection → SiLU(gate) * up → down projection → residual

#include "lightllm/tensor.h"
#include "lightllm/model/model_config.h"

namespace lightllm {
namespace model {

struct BlockWeights {
    // Attention
    Tensor q_weight, q_bias;  // [D,D] and [D]
    Tensor k_weight, k_bias;  // [D_kv,D] and [D_kv]
    Tensor v_weight, v_bias;  // [D_kv,D] and [D_kv]
    Tensor o_weight;          // [D,D]
    Tensor attn_norm;         // [D]

    // MLP
    Tensor gate_weight, up_weight;  // [inter,D], [inter,D]
    Tensor down_weight;             // [D,inter]
    Tensor mlp_norm;                // [D]
};

class TransformerBlock {
public:
    TransformerBlock(const BlockWeights& w, const ModelConfig& cfg);

    /// Forward pass. Modifies kv_cache in-place, returns output.
    /// x: [num_tokens, D]
    /// positions: [num_tokens] token positions for RoPE
    /// kv_cache: pre-allocated [2, num_tokens, D_kv] for K and V
    Tensor forward(const Tensor& x, const Tensor& positions,
                   Tensor& k_cache, Tensor& v_cache);

private:
    BlockWeights w_;
    ModelConfig cfg_;
    int D_, D_kv_, num_heads_, num_kv_heads_, head_dim_;
};

}  // namespace model
}  // namespace lightllm
