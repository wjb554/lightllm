/// Softmax — 3-kernel GPU with proper cross-warp reduction.
/// Kernels template on scalar_t (float/half/__nv_bfloat16) but accumulate in float
/// for numerical stability. The public API dispatches via DISPATCH_FLOAT_TYPES.

#include "lightllm/ops/softmax.h"
#include "lightllm/ops/dispatch.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace lightllm {
namespace ops {

// Kernel 1: compute per-row max, store in row_buf (always float).
template<typename T>
__global__ void sm_max_kernel(float* row_buf, const T* x, int cols) {
    int row = blockIdx.x, tid = threadIdx.x;
    __shared__ float warp_max[32];
    int wid = tid / 32, lid = tid % 32;
    float mx = -1e38f;
    for (int i = tid; i < cols; i += blockDim.x)
        mx = fmaxf(mx, float(x[row * cols + i]));
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    if (lid == 0) warp_max[wid] = mx;
    __syncthreads();
    if (tid == 0) {
        int nw = blockDim.x / 32;
        for (int w = 1; w < nw; w++) mx = fmaxf(mx, warp_max[w]);
        row_buf[row] = mx;
    }
}

// Kernel 2: exp(x - max) in-place, accumulate sum in float, store sum in row_buf.
template<typename T>
__global__ void sm_apply_kernel(T* x, float* row_buf, int cols) {
    int row = blockIdx.x, tid = threadIdx.x;
    __shared__ float warp_sum[32];
    int wid = tid / 32, lid = tid % 32;
    float mx = row_buf[row], sm = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = expf(float(x[row * cols + i]) - mx);
        x[row * cols + i] = T(v);
        sm += v;
    }
    for (int o = 16; o > 0; o >>= 1)
        sm += __shfl_down_sync(0xffffffff, sm, o);
    if (lid == 0) warp_sum[wid] = sm;
    __syncthreads();
    if (tid == 0) {
        int nw = blockDim.x / 32;
        for (int w = 1; w < nw; w++) sm += warp_sum[w];
        row_buf[row] = sm;
    }
}

// Kernel 3: divide by sum in-place.
template<typename T>
__global__ void sm_norm_kernel(T* x, const float* row_buf, int cols) {
    int row = blockIdx.x, tid = threadIdx.x;
    float inv = 1.0f / (row_buf[row] + 1e-8f);
    for (int i = tid; i < cols; i += blockDim.x)
        x[row * cols + i] = T(float(x[row * cols + i]) * inv);
}

Tensor softmax(const Tensor& x) {
    int rows = x.numel() / x.size(-1), cols = x.size(-1);
    int blk = (cols < 256) ? (1 << ((int)ceilf(log2f((float)cols)))) : 256;
    if (blk < 32) blk = 32;

    // Output tensor (same shape/dtype as input). Operate in-place on it.
    Tensor out(x.shape(), x.dtype(), Device::CUDA);
    cudaMemcpy(out.raw(), x.raw(), rows * cols * dtype_size(x.dtype()),
               cudaMemcpyDeviceToDevice);

    // Per-row scratch buffer in float (size = rows).
    Tensor row_buf({rows}, DType::F32, Device::CUDA);

    DISPATCH_FLOAT_TYPES(x.dtype(), "softmax", {
        sm_max_kernel<scalar_t><<<rows, blk>>>(row_buf.data<float>(), out.data<scalar_t>(), cols);
        sm_apply_kernel<scalar_t><<<rows, blk>>>(out.data<scalar_t>(), row_buf.data<float>(), cols);
        sm_norm_kernel<scalar_t><<<rows, blk>>>(out.data<scalar_t>(), row_buf.data<float>(), cols);
    });

    return std::move(out);
}

}  // namespace ops
}  // namespace lightllm
