#pragma once
/// Minimal tokenizer for HuggingFace tokenizer.json format.
/// Decode-only: token IDs → readable text.
/// For encode (text→tokens), use HuggingFace Python tokenizer as a pre-processing step.

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lightllm {
namespace tokenizer {

class Tokenizer {
public:
    /// Load from HuggingFace tokenizer.json.
    /// Download: https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer.json
    explicit Tokenizer(const std::string& path);

    /// Decode token IDs to a UTF-8 string.
    std::string decode(const std::vector<int>& token_ids) const;

    /// Get the vocab size.
    int vocab_size() const { return (int)id_to_token_.size(); }

    /// Get the token string for a given ID.
    const std::string& id_to_str(int id) const;

    /// Encode a string to token IDs (simplified: whitespace split + lookup).
    /// For production use, pre-tokenize with Python HuggingFace tokenizer.
    std::vector<int> encode(const std::string& text) const;

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::pair<std::string,std::string>> merges_; // BPE merges
    std::unordered_map<std::string, unsigned char> byte_decoder_;
};

}  // namespace tokenizer
}  // namespace lightllm
