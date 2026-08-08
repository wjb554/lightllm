#include "lightllm/tensor.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

namespace lightllm {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static void cuda_check(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(ctx) + ": " + cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// construction / destruction
// ---------------------------------------------------------------------------
Tensor::Tensor() = default;

Tensor::Tensor(std::vector<int> shape, DType dtype, Device device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device) {
    allocate();
}

Tensor::~Tensor() {
    if (owns_data_) deallocate();
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      device_(other.device_),
      owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (owns_data_) deallocate();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        owns_data_ = other.owns_data_;
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// memory
// ---------------------------------------------------------------------------
void Tensor::allocate() {
    size_t bytes = nbytes();
    if (bytes == 0) return;

    if (device_ == Device::CUDA) {
        cuda_check(cudaMalloc(&data_, bytes), "cudaMalloc");
    } else {
        data_ = std::malloc(bytes);
        if (!data_) throw std::bad_alloc();
    }
}

void Tensor::deallocate() {
    if (!data_) return;
    if (device_ == Device::CUDA) {
        cudaFree(data_);
    } else {
        std::free(data_);
    }
    data_ = nullptr;
}

void Tensor::copy_from(const void* src, size_t bytes) {
    if (bytes > nbytes()) {
        throw std::runtime_error("copy_from: source larger than tensor");
    }
    if (device_ == Device::CUDA) {
        cuda_check(
            cudaMemcpy(data_, src, bytes, cudaMemcpyHostToDevice),
            "copy_from H2D");
    } else {
        std::memcpy(data_, src, bytes);
    }
}

void Tensor::copy_to(void* dst, size_t bytes) const {
    if (bytes > nbytes()) {
        throw std::runtime_error("copy_to: request larger than tensor");
    }
    if (device_ == Device::CUDA) {
        cuda_check(
            cudaMemcpy(dst, data_, bytes, cudaMemcpyDeviceToHost),
            "copy_to D2H");
    } else {
        std::memcpy(dst, data_, bytes);
    }
}

// ---------------------------------------------------------------------------
// metadata
// ---------------------------------------------------------------------------
size_t Tensor::numel() const {
    size_t n = 1;
    for (int d : shape_) n *= d;
    return n;
}

size_t Tensor::nbytes() const {
    return numel() * dtype_size(dtype_);
}

Tensor Tensor::view(std::vector<int> new_shape) const {
    size_t new_numel = 1;
    for (int d : new_shape) new_numel *= d;
    if (new_numel != numel()) {
        throw std::runtime_error(
            "view: shape has different numel ("
            + std::to_string(new_numel) + " vs " + std::to_string(numel()) + ")");
    }
    Tensor v;
    v.data_ = data_;
    v.shape_ = std::move(new_shape);
    v.dtype_ = dtype_;
    v.device_ = device_;
    v.owns_data_ = false;  // view — does not own
    return v;
}

void Tensor::reshape_inplace(std::vector<int> new_shape) {
    size_t new_numel = 1;
    for (int d : new_shape) new_numel *= d;
    if (new_numel != numel()) {
        throw std::runtime_error(
            "reshape_inplace: shape has different numel ("
            + std::to_string(new_numel) + " vs " + std::to_string(numel()) + ")");
    }
    shape_ = std::move(new_shape);
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------
void Tensor::fill_zeros() {
    if (!data_) return;
    size_t bytes = nbytes();
    if (device_ == Device::CUDA) {
        cuda_check(cudaMemset(data_, 0, bytes), "cudaMemset");
    } else {
        std::memset(data_, 0, bytes);
    }
}

std::string Tensor::to_string(int max_elems) const {
    if (!data_) return "<empty>";

    size_t n = numel();
    size_t show = std::min(static_cast<size_t>(max_elems), n);
    std::string s = "Tensor(";
    for (size_t i = 0; i < shape_.size(); i++) {
        s += std::to_string(shape_[i]);
        if (i + 1 < shape_.size()) s += "×";
    }
    s += ", " + std::string(dtype_name(dtype_)) + ", "
       + (device_ == Device::CUDA ? "cuda" : "cpu") + ")\n[";

    // Read first `show` elements to host
    if (dtype_ == DType::F32) {
        std::vector<float> tmp(show);
        copy_to(tmp.data(), show * sizeof(float));
        for (size_t i = 0; i < show; i++) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.4f", tmp[i]);
            if (i > 0) s += ", ";
            s += buf;
        }
    } else if (dtype_ == DType::F16 || dtype_ == DType::BF16) {
    } else {
        s += "...";
    }

    if (show < n) s += ", ...";
    s += "]";
    return s;
}

}  // namespace lightllm
