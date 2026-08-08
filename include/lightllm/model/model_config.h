#pragma once
/// Model configuration — extracted from HuggingFace config.json.
/// Supports Llama and Qwen2 architectures (they're structurally identical).

#include <string>
#include <vector>
#include "lightllm/tensor.h"

namespace lightllm {
namespace model {

struct ModelConfig {
    std::string architecture;        // "LlamaForCausalLM" or "Qwen2ForCausalLM"
    int hidden_size = 0;             // d_model
    int intermediate_size = 0;       // MLP hidden dim
    int num_attention_heads = 0;     // Q heads
    int num_key_value_heads = 0;     // KV heads (GQA)
    int num_hidden_layers = 0;       // transformer blocks
    int head_dim = 0;                // hidden_size / num_attention_heads
    float rms_norm_eps = 1e-6f;
    int vocab_size = 0;
    int max_position_embeddings = 0;
    float rope_theta = 10000.0f;
    bool tie_word_embeddings = false;

    // Derived
    int kv_groups() const { return num_attention_heads / num_key_value_heads; }
    size_t total_params() const;
    std::string to_string() const;
};

/// Parse a HuggingFace config.json into ModelConfig.
ModelConfig parse_config(const std::string& json_path);

}  // namespace model
}  // namespace lightllm
