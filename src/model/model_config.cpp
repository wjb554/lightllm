/// HuggingFace config.json parser (manual, no JSON library dependency).

#include "lightllm/model/model_config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lightllm {
namespace model {

// Extract integer value for a given key from JSON text
static int json_int(const std::string& json, const std::string& key) {
    size_t p = json.find("\"" + key + "\"");
    if (p == std::string::npos) return 0;
    p = json.find(':', p);
    if (p == std::string::npos) return 0;
    p++;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) p++;
    std::string val;
    while (p < json.size() && (json[p] >= '0' && json[p] <= '9' || json[p] == '-' || json[p] == '.' || json[p] == 'e')) {
        val += json[p]; p++;
    }
    return val.empty() ? 0 : std::stoi(val);
}

static float json_float(const std::string& json, const std::string& key) {
    size_t p = json.find("\"" + key + "\"");
    if (p == std::string::npos) return 0.f;
    p = json.find(':', p);
    if (p == std::string::npos) return 0.f;
    p++;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) p++;
    std::string val;
    while (p < json.size() && (json[p] >= '0' && json[p] <= '9' || json[p] == '-' || json[p] == '.' || json[p] == 'e')) {
        val += json[p]; p++;
    }
    return val.empty() ? 0.f : std::stof(val);
}

static bool json_bool(const std::string& json, const std::string& key) {
    size_t p = json.find("\"" + key + "\"");
    if (p == std::string::npos) return false;
    p = json.find(':', p);
    if (p == std::string::npos) return false;
    p++;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) p++;
    return json.substr(p, 4) == "true";
}

static std::string json_str(const std::string& json, const std::string& key) {
    size_t p = json.find("\"" + key + "\"");
    if (p == std::string::npos) return "";
    p = json.find(':', p);
    if (p == std::string::npos) return "";
    p = json.find('"', p);
    if (p == std::string::npos) return "";
    size_t e = json.find('"', p + 1);
    return json.substr(p + 1, e - p - 1);
}

ModelConfig parse_config(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f) throw std::runtime_error("Cannot open config: " + json_path);
    std::stringstream ss; ss << f.rdbuf();
    std::string json = ss.str();

    ModelConfig cfg;
    cfg.architecture = json_str(json, "architectures");
    // "architectures" is an array like ["Qwen2ForCausalLM"] — extract from array
    if (cfg.architecture.empty()) {
        size_t pa = json.find("\"architectures\"");
        if (pa != std::string::npos) {
            pa = json.find('[', pa);
            size_t pe = json.find(']', pa);
            std::string arr = json.substr(pa + 1, pe - pa - 1);
            // extract first quoted string
            size_t q1 = arr.find('"');
            if (q1 != std::string::npos) {
                size_t q2 = arr.find('"', q1 + 1);
                cfg.architecture = arr.substr(q1 + 1, q2 - q1 - 1);
            }
        }
    }

    cfg.hidden_size = json_int(json, "hidden_size");
    cfg.intermediate_size = json_int(json, "intermediate_size");
    cfg.num_attention_heads = json_int(json, "num_attention_heads");
    cfg.num_key_value_heads = json_int(json, "num_key_value_heads");
    cfg.num_hidden_layers = json_int(json, "num_hidden_layers");
    cfg.vocab_size = json_int(json, "vocab_size");
    cfg.max_position_embeddings = json_int(json, "max_position_embeddings");

    cfg.rms_norm_eps = json_float(json, "rms_norm_eps");
    cfg.rope_theta = json_float(json, "rope_theta");
    cfg.tie_word_embeddings = json_bool(json, "tie_word_embeddings");

    if (cfg.num_attention_heads > 0)
        cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;

    if (cfg.num_key_value_heads == 0)
        cfg.num_key_value_heads = cfg.num_attention_heads;

    return cfg;
}

size_t ModelConfig::total_params() const {
    size_t n = 0;
    // Embedding
    n += static_cast<size_t>(vocab_size) * hidden_size;
    // Per layer: attn QKV+O + MLP gate+up+down + 2 RMS norms
    for (int i = 0; i < num_hidden_layers; i++) {
        size_t head_dim_kv = hidden_size / num_attention_heads;
        n += hidden_size * hidden_size;                          // Q
        n += num_key_value_heads * head_dim_kv * hidden_size * 2; // K + V
        n += hidden_size * hidden_size;                          // O
        n += hidden_size * intermediate_size;                    // gate
        n += hidden_size * intermediate_size;                    // up
        n += intermediate_size * hidden_size;                    // down
        n += hidden_size * 2;                                     // 2 RMS norms
    }
    // Final norm + lm_head (if not tied)
    n += hidden_size;
    if (!tie_word_embeddings) n += static_cast<size_t>(vocab_size) * hidden_size;
    return n;
}

std::string ModelConfig::to_string() const {
    char buf[512];
    snprintf(buf, sizeof(buf),
        "%s: %d layers, d=%d, inter=%d, heads=%d (kv=%d), vocab=%d\n"
        "  head_dim=%d, groups=%d, rope_theta=%.1f, tied=%d, params=%.1fM",
        architecture.c_str(), num_hidden_layers, hidden_size, intermediate_size,
        num_attention_heads, num_key_value_heads, vocab_size,
        head_dim, kv_groups(), rope_theta, tie_word_embeddings,
        total_params() / 1e6);
    return std::string(buf);
}

}  // namespace model
}  // namespace lightllm
