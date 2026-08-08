/// RMSNorm — simplified two-kernel version for debuggability.
/// Kernel 1: compute sum(x^2) per row. Kernel 2: normalize.

#include "lightllm/ops/norm.h"
#include <cuda_runtime.h>
#include "lightllm/ops/dispatch.h"

namespace lightllm {
namespace ops {

template <typename T>
__global__ void rms_sum_sq_kernel(
    float* __restrict__ row_rms,       // [num_tokens]
    const T* __restrict__ x,           // [num_tokens, hidden_size]
    int num_tokens, int hidden_size) {

    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = float(x[row * hidden_size + i]);
        sum_sq += val * val;
    }
    // Warp reduce
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);
    // First thread in each warp writes to shared, first warp lane writes to global
    __shared__ float smem[32];
    if (tid % 32 == 0) smem[tid / 32] = sum_sq;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < blockDim.x / 32; w++) total += smem[w];
        row_rms[row] = total;
    }
}

template <typename T>
__global__ void rms_apply_kernel(
    T* __restrict__ out,
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const float* __restrict__ row_rms, // sum(x^2) per row
    int num_tokens, int hidden_size, float eps) {

    int row = blockIdx.x;
    int tid = threadIdx.x;
    float rms = sqrtf(row_rms[row] / float(hidden_size) + eps);
    float inv_rms = 1.0f / rms;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = float(x[row * hidden_size + i]);
        float w = float(weight[i]);
        out[row * hidden_size + i] = T(val * inv_rms * w);
    }
}

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    int N = x.size(0), D = x.size(1);
    if (weight.size(0) != D)
        throw std::runtime_error("rms_norm: weight dim mismatch");

    Tensor out(x.shape(), x.dtype(), Device::CUDA);
    Tensor row_rms({N}, DType::F32, Device::CUDA);

    int block = 256, grid = N;

    DISPATCH_FLOAT_HALF(x.dtype(), "rms_norm", {
        rms_sum_sq_kernel<scalar_t><<<grid, block>>>(
            row_rms.data<float>(), x.data<scalar_t>(), N, D);
        rms_apply_kernel<scalar_t><<<grid, block>>>(
            out.data<scalar_t>(), x.data<scalar_t>(),
            weight.data<scalar_t>(), row_rms.data<float>(), N, D, eps);
    });

    return std::move(out);
}

}  // namespace ops
}  // namespace lightllm
