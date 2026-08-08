#pragma once
/// GPU lm_head — dot-product hidden state against embedding weight matrix.
///
/// Computes:  logits[v] = sum_d( hidden[d] * embed_weight[v * D + d] )
///
/// Hidden  [D] or [1,D]  fp32 GPU
/// Embed   [vocab, D]    fp32 GPU
/// Returns [vocab]        fp32 GPU logits
///
/// This eliminates the ~520 MB GPU-to-CPU copy of embed_weight that the
/// old CPU lm_head path performed on every decode token.  With D=896 and
/// vocab=151936 the kernel reads ~135 M elements = ~540 MB from HBM, which
/// at ~700 GB/s (A5000-class) takes < 1 ms — compared to ~21 ms for the
/// PCIe round-trip the CPU path incurred.
///
/// Architecture:
///   Grid:  ceil(vocab / 256) blocks of 256 threads
///   Phase 1 — cooperative load:  hidden[D] is loaded into
///            __shared__ float sh_h[D] (896 elements = 3.5 KB).
///   Phase 2 — dot products:  each thread strides over its assigned
///            vocab rows.  Thread `tid = blockIdx.x * 256 + threadIdx.x`
///            handles rows {tid, tid+stride, tid+2*stride, …} where
///            stride = gridDim.x * 256.  For each row it computes
///            sum_d sh_h[d] * embed[row * D + d] and writes the result
///            to logits[row].
///
/// Memory access:
///   - Within one row the inner loop reads D contiguous floats — full
///     cache-line utilisation.
///   - Adjacent threads in a warp read rows that are D elements apart
///     in memory, so the first-element accesses are not coalesced, but
///     the L2 cache absorbs this gracefully for the memory-bound regime.
///   - The shared-memory broadcast of hidden[D] turns 151936 redundant
///     HBM reads of hidden into a single cooperative load.
///
/// If D ever exceeds 1024, switch the kernel to use dynamic shared memory
/// (extern __shared__ float sh_h[]) with the third launch parameter set
/// to D * sizeof(float).

#include "lightllm/tensor.h"

namespace lightllm {
namespace ops {

/// Compute full-vocabulary logits on GPU.
///
/// \param hidden       [D] or [1, D]  fp32 on CUDA device — final-layer
///                     hidden state after RMSNorm.
/// \param embed_weight [vocab, D]     fp32 on CUDA device — the unembedding
///                     matrix (typically tied with input embeddings).
/// \return             Tensor of shape [vocab], dtype fp32, on CUDA device.
///                     Caller copies the result to CPU if sampling runs on
///                     the host (~608 KB for vocab=151936).
Tensor lm_head_logits(const Tensor& hidden, const Tensor& embed_weight);

}  // namespace ops
}  // namespace lightllm
