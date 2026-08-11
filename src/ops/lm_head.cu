/// GPU lm_head kernel — see include/lightllm/ops/lm_head.h for architecture docs.
#include "lightllm/ops/lm_head.h"
#include "lightllm/ops/dispatch.h"
#include "lightllm/ops/gemm.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace lightllm {
namespace ops {

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
// Block: 256 threads (1-D)
// Grid:  min(ceil_div(vocab, 256), 65535)
// Dynamic shared memory: D * sizeof(float) bytes, passed as kernel<<<..., smem>>>
//
// Phase 1 — cooperative load of hidden[D] into __shared__ hs[].
//          256 threads load D elements in ceil(D/256) rounds.
// Phase 2 — strided dot products.  Each thread handles rows
//          {tid, tid+stride, tid+2*stride, …} where stride = gridDim.x * 256.
//          For each row, dot = sum_d hs[d] * embed[row * D + d].
//          Reads within a row are fully contiguous (cache-line friendly).
// ---------------------------------------------------------------------------
__global__ void lm_head_logits_kernel(
    float* __restrict__ logits,      // [vocab] output
    const float* __restrict__ h,     // [D] input hidden state
    const float* __restrict__ embed, // [vocab, D] weight, row-major
    int D,                           // hidden / embedding dimension
    int vocab)                       // vocabulary size
{
    extern __shared__ float hs[];

    // Phase 1: cooperative load of hidden[D] into shared memory.
    // 256 threads load D elements — 4 rounds for D=896.
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        hs[i] = h[i];
    __syncthreads();

    // Phase 2: strided dot products.
    // Each thread is responsible for ceil(vocab / stride) rows.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int v = tid; v < vocab; v += stride) {
        const float* row = embed + (size_t)v * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d)
            dot += hs[d] * row[d];
        logits[v] = dot;
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
Tensor lm_head_logits(const Tensor& h, const Tensor& embed) {
    // Accept hidden as [D] or [1, D] — last dimension is always D.
    int D = h.size(h.dims() - 1);
    int vocab = embed.size(0);

    if (h.dtype() == DType::F16 && embed.dtype() == DType::F16) {
        return gemm_f16f32(h, embed, true);
    }

    if (embed.size(1) != D) {
        throw std::runtime_error(
            "lm_head_logits: dim mismatch — hidden D=" + std::to_string(D) +
            ", embed D=" + std::to_string(embed.size(1)));
    }

    Tensor logits({vocab}, DType::F32, Device::CUDA);

    static constexpr int BLOCK = 256;
    int grid = std::min(ceil_div(vocab, BLOCK), 65535);
    size_t smem = D * sizeof(float);

    lm_head_logits_kernel<<<dim3(grid), dim3(BLOCK), smem>>>(
        logits.data<float>(),
        h.data<float>(),
        embed.data<float>(),
        D,
        vocab);

    return logits;
}

}  // namespace ops
}  // namespace lightllm
