/// Safetensors loader — manual JSON parsing to avoid external dependencies.
///
/// The safetensors header is a flat JSON object:
///   {"tensor_name": {"dtype": "BF16", "shape": [a,b], "data_offsets": [x,y]}, ...}
///
/// We parse it with simple string operations (no JSON library needed).

#include "lightllm/model/model_loader.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace lightllm {
namespace model {

// --- helpers ---------------------------------------------------------------

static DType parse_dtype(const std::string& s) {
    if (s == "\"F32\"") return DType::F32;
    if (s == "\"F16\"") return DType::F16;
    if (s == "\"BF16\"") return DType::BF16;
    throw std::runtime_error("Unknown safetensors dtype: " + s);
}

static std::vector<int> parse_shape(const std::string& s) {
    // s is like "[151936,896]" or "[896]"
    std::vector<int> shape;
    size_t i = s.find('[');
    if (i == std::string::npos) return shape;
    i++;
    while (i < s.size() && s[i] != ']') {
        while (i < s.size() && (s[i] == ' ' || s[i] == ',')) i++;
        if (i >= s.size() || s[i] == ']') break;
        size_t end = i;
        while (end < s.size() && s[end] >= '0' && s[end] <= '9') end++;
        shape.push_back(std::stoi(s.substr(i, end - i)));
        i = end;
    }
    return shape;
}

// --- SafetensorsLoader -----------------------------------------------------

SafetensorsLoader::SafetensorsLoader(const std::string& path) : path_(path) {
    // Read the entire file to get the header
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);

    // Read header length (8 bytes, little-endian uint64)
    char len_buf[8];
    file.read(len_buf, 8);
    uint64_t header_len = *reinterpret_cast<uint64_t*>(len_buf);

    // Read header JSON
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    data_offset_ = 8 + header_len;

    // Parse: find each "tensor_name": { ... }
    size_t pos = header.find('{');  // first '{' after __metadata__
    if (pos == std::string::npos) throw std::runtime_error("Invalid safetensors header");

    // Move past the opening '{' and any __metadata__ entry
    if (header.find("\"__metadata__\"") != std::string::npos) {
        // Skip the __metadata__ object
        pos = header.find("},\"", pos);
        if (pos == std::string::npos) pos = header.find("}\"", pos);
        if (pos != std::string::npos) pos += 2;  // skip "},"
    } else {
        pos = 1;
    }

    // Parse each tensor entry: "name": {"dtype":...,"shape":...,"data_offsets":...}
    while (pos < header.size()) {
        if (header[pos] == '}' || header[pos] == '\0') break;

        // Find tensor name
        size_t name_start = header.find('"', pos);
        if (name_start == std::string::npos) break;
        size_t name_end = header.find('"', name_start + 1);
        std::string name = header.substr(name_start + 1, name_end - name_start - 1);
        if (name.empty() || name == "__metadata__") { pos = name_end + 1; continue; }

        // Find the value object: { ... }
        size_t obj_start = header.find('{', name_end);
        size_t obj_end = header.find('}', obj_start);
        std::string obj = header.substr(obj_start, obj_end - obj_start + 1);

        TensorInfo info;

        // Parse dtype
        size_t dt_pos = obj.find("\"dtype\"");
        size_t dt_colon = obj.find(':', dt_pos);
        size_t dt_start = obj.find('"', dt_colon);
        size_t dt_end = obj.find('"', dt_start + 1);
        info.dtype = parse_dtype(obj.substr(dt_start, dt_end - dt_start + 1));

        // Parse shape
        size_t sh_pos = obj.find("\"shape\"");
        size_t sh_colon = obj.find(':', sh_pos);
        size_t sh_start = obj.find('[', sh_colon);
        size_t sh_end = obj.find(']', sh_start);
        info.shape = parse_shape(obj.substr(sh_start, sh_end - sh_start + 1));

        // Parse data_offsets
        size_t do_pos = obj.find("\"data_offsets\"");
        size_t do_colon = obj.find(':', do_pos);
        size_t do_start = obj.find('[', do_colon);
        size_t do_comma = obj.find(',', do_start);
        size_t do_end = obj.find(']', do_comma);
        info.offset_begin = std::stoull(obj.substr(do_start + 1, do_comma - do_start - 1));
        info.offset_end = std::stoull(obj.substr(do_comma + 1, do_end - do_comma - 1));

        tensors_[name] = info;
        pos = obj_end + 2;  // skip "},"
    }
}

const TensorInfo* SafetensorsLoader::get_info(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

Tensor SafetensorsLoader::load_tensor(const std::string& name) const {
    auto* info = get_info(name);
    if (!info) throw std::runtime_error("Tensor not found: " + name);

    Tensor t(info->shape, info->dtype, Device::CPU);
    load_into(name, t);
    return t;
}

void SafetensorsLoader::load_into(const std::string& name, Tensor& dst) const {
    auto* info = get_info(name);
    if (!info) throw std::runtime_error("Tensor not found: " + name);

    if (dst.shape() != info->shape || dst.dtype() != info->dtype)
        throw std::runtime_error("Shape/dtype mismatch for: " + name);

    size_t tensor_bytes = info->offset_end - info->offset_begin;
    if (tensor_bytes != dst.nbytes())
        throw std::runtime_error("Byte size mismatch for: " + name);

    // Read raw data from file into CPU tensor
    std::ifstream file(path_, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path_);

    // For large tensors, read in chunks
    file.seekg(data_offset_ + info->offset_begin);
    std::vector<char> buf(tensor_bytes);
    file.read(buf.data(), tensor_bytes);

    // Copy from CPU buffer to Tensor (whether CPU or GPU)
    if (dst.is_cuda()) {
        // If destination is GPU, copy via pinned memory for speed
        dst.copy_from(buf.data(), tensor_bytes);
    } else {
        std::memcpy(dst.raw(), buf.data(), tensor_bytes);
    }
}

std::vector<std::string> SafetensorsLoader::tensor_names() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : tensors_) names.push_back(name);
    return names;
}

}  // namespace model
}  // namespace lightllm
