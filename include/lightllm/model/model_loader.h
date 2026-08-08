#pragma once
/// Safetensors file loader — reads HuggingFace model weights into GPU Tensors.
///
/// Format: [8-byte header_len (uint64)] [JSON header] [raw tensor data]
/// Each tensor in the header has: dtype, shape, data_offsets [begin, end].

#include <string>
#include <unordered_map>
#include <vector>

#include "lightllm/tensor.h"

namespace lightllm {
namespace model {

struct TensorInfo {
    DType dtype;
    std::vector<int> shape;
    size_t offset_begin;
    size_t offset_end;
};

class SafetensorsLoader {
public:
    /// Open a safetensors file and parse the header.
    explicit SafetensorsLoader(const std::string& path);

    /// Get metadata for a tensor by name.
    const TensorInfo* get_info(const std::string& name) const;

    /// Load one tensor into GPU memory (allocates and copies).
    Tensor load_tensor(const std::string& name) const;

    /// Load one tensor into a pre-allocated Tensor (must match shape/dtype).
    void load_into(const std::string& name, Tensor& dst) const;

    /// List all tensor names in the file.
    std::vector<std::string> tensor_names() const;

    /// Number of tensors.
    size_t num_tensors() const { return tensors_.size(); }

private:
    std::string path_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    size_t data_offset_;  // byte offset to the start of raw data
};

}  // namespace model
}  // namespace lightllm
