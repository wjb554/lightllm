#pragma once
/// LightLLM Tensor — GPU/CPU tensor with RAII memory management.
///
/// Row-major contiguous storage, matching PyTorch/HuggingFace convention.
/// Supports fp32, fp16, bf16, and int32 dtypes on CUDA and CPU devices.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lightllm {

enum class DType : uint8_t { F32 = 0, F16 = 1, BF16 = 2, I32 = 3 };

enum class Device : uint8_t { CPU = 0, CUDA = 1 };

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::F32:  return "f32";
        case DType::F16:  return "f16";
        case DType::BF16: return "bf16";
        case DType::I32:  return "i32";
    }
    return "unknown";
}

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::I32:  return 4;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Tensor — owns (or views) a contiguous buffer on CPU or GPU
// ---------------------------------------------------------------------------
class Tensor {
public:
    // --- construction ---
    Tensor();  // empty / null tensor
    Tensor(std::vector<int> shape, DType dtype, Device device);
    ~Tensor();

    // Movable but not copyable (unique ownership of data)
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // --- data access ---
    template <typename T>
    T* data() { return static_cast<T*>(data_); }

    template <typename T>
    const T* data() const { return static_cast<const T*>(data_); }

    void* raw() { return data_; }
    const void* raw() const { return data_; }

    void copy_from(const void* src, size_t bytes);
    void copy_to(void* dst, size_t bytes) const;

    // --- metadata ---
    const std::vector<int>& shape() const { return shape_; }
    int dims() const { return static_cast<int>(shape_.size()); }
    int size(int dim) const { return shape_[(dim<0)?dims()+dim:dim]; }

    size_t numel() const;
    size_t nbytes() const;

    DType dtype() const { return dtype_; }
    Device device() const { return device_; }

    bool is_cpu() const { return device_ == Device::CPU; }
    bool is_cuda() const { return device_ == Device::CUDA; }
    bool defined() const { return data_ != nullptr; }

    // Reshape (must preserve numel); returns new view (no copy).
    // The returned Tensor does NOT own the data.
    Tensor view(std::vector<int> new_shape) const;

    // Reshape IN-PLACE (must preserve numel). Safe for self-assignment
    // patterns like `t = t.view(...)` which would otherwise double-free.
    void reshape_inplace(std::vector<int> new_shape);

    // --- I/O helpers ---
    /// Fill with zeros on device.
    void fill_zeros();

    /// Print first N elements (host-side copy).
    std::string to_string(int max_elems = 16) const;

private:
    void* data_ = nullptr;
    std::vector<int> shape_;
    DType dtype_ = DType::F32;
    Device device_ = Device::CPU;
    bool owns_data_ = true;

    void allocate();
    void deallocate();
};

}  // namespace lightllm
