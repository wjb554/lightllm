/// GPU argmax — block-level reduction to find max-index.
/// Single block (256 threads), strided scan over vocab_size elements.
#include "lightllm/ops/argmax.h"
#include "lightllm/ops/dispatch.h"
#include <cuda_runtime.h>
#include <cfloat>

namespace lightllm {
namespace ops {

template<typename T>
__global__ void argmax_kernel(int* out_idx, const T* x, int N) {
    __shared__ float s_vals[256];
    __shared__ int   s_idxs[256];

    int tid = threadIdx.x;
    float best_val = -FLT_MAX;
    int   best_idx = 0;

    // Strided scan over vocabulary
    for (int i = tid; i < N; i += blockDim.x) {
        float v = static_cast<float>(x[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    // Store per-thread best into shared memory
    s_vals[tid] = best_val;
    s_idxs[tid] = best_idx;
    __syncthreads();

    // Tree reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[0] = s_idxs[0];
    }
}

int argmax(const Tensor& logits) {
    int N = static_cast<int>(logits.numel());
    Tensor out_idx({1}, DType::I32, Device::CUDA);

    DISPATCH_FLOAT_TYPES(logits.dtype(), "argmax", {
        argmax_kernel<scalar_t><<<1, 256>>>(
            out_idx.data<int>(),
            logits.data<scalar_t>(),
            N);
    });

    int result;
    out_idx.copy_to(&result, sizeof(int));
    return result;
}

}  // namespace ops
}  // namespace lightllm
