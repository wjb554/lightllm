#include "lightllm/ops/elementwise.h"
#include <cuda_runtime.h>
#include "lightllm/ops/dispatch.h"

namespace lightllm {
namespace ops {

template <typename T>
__global__ void add_kernel(T* out, const T* a, const T* b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = a[idx] + b[idx];
}

template <typename T>
__global__ void mul_kernel(T* out, const T* a, const T* b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = a[idx] * b[idx];
}

template <typename T>
__global__ void scale_kernel(T* out, const T* x, float factor, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = T(float(x[idx]) * factor);
}

template <typename T>
__global__ void silu_kernel(T* x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = float(x[idx]);
        x[idx] = T(val / (1.0f + expf(-val)));
    }
}

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("add: shape mismatch");
    if (a.dtype() != b.dtype()) throw std::runtime_error("add: dtype mismatch");
    Tensor out(a.shape(), a.dtype(), Device::CUDA);
    int N = int(a.numel()), block = 256, grid = ceil_div(N, block);
    DISPATCH_FLOAT_TYPES(a.dtype(), "add", {
        add_kernel<scalar_t><<<grid, block>>>(
            out.data<scalar_t>(), a.data<scalar_t>(), b.data<scalar_t>(), N);
    });
    return std::move(out);
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("mul: shape mismatch");
    if (a.dtype() != b.dtype()) throw std::runtime_error("mul: dtype mismatch");
    Tensor out(a.shape(), a.dtype(), Device::CUDA);
    int N = int(a.numel()), block = 256, grid = ceil_div(N, block);
    DISPATCH_FLOAT_TYPES(a.dtype(), "mul", {
        mul_kernel<scalar_t><<<grid, block>>>(
            out.data<scalar_t>(), a.data<scalar_t>(), b.data<scalar_t>(), N);
    });
    return std::move(out);
}

Tensor scale(const Tensor& x, float factor) {
    Tensor out(x.shape(), x.dtype(), Device::CUDA);
    int N = int(x.numel()), block = 256, grid = ceil_div(N, block);
    DISPATCH_FLOAT_TYPES(x.dtype(), "scale", {
        scale_kernel<scalar_t><<<grid, block>>>(
            out.data<scalar_t>(), x.data<scalar_t>(), factor, N);
    });
    return std::move(out);
}

Tensor silu(const Tensor& x) {
    Tensor out(x.shape(), x.dtype(), Device::CUDA);
    int N = int(x.numel()), block = 256, grid = ceil_div(N, block);
    DISPATCH_FLOAT_TYPES(x.dtype(), "silu", {
        cudaMemcpy(out.data<scalar_t>(), x.data<scalar_t>(), x.nbytes(), cudaMemcpyDeviceToDevice);
        silu_kernel<scalar_t><<<grid, block>>>(out.data<scalar_t>(), N);
    });
    return std::move(out);
}

void silu_inplace(Tensor& x) {
    int N = int(x.numel()), block = 256, grid = ceil_div(N, block);
    DISPATCH_FLOAT_TYPES(x.dtype(), "silu_inplace", {
        silu_kernel<scalar_t><<<grid, block>>>(x.data<scalar_t>(), N);
    });
}

}  // namespace ops
}  // namespace lightllm
