#pragma once
/// PagedAttention CUDA kernels — decode and prefill variants.
///
/// Decode:  one block per (token, Q-head) pair, iterating over KV blocks as
///          each batch entry contributes exactly one new token.
/// Prefill: one call handles a P-token chunk for a single request, reading
///          K/V directly from paged blocks so no contiguous reconstruction
///          tensors or per-position cudaMemcpy are required.

#include "lightllm/tensor.h"

namespace lightllm {
namespace kv_cache {

// ---------------------------------------------------------------------------
// Decode PagedAttention — batched (num_tokens, 1 token each)
// ---------------------------------------------------------------------------

/// @param q            [num_tokens, num_q_heads, head_dim]
/// @param block_table  flat array of physical block IDs, size
///                     [num_seqs * max_blocks_per_seq]
/// @param max_blocks_per_seq  max blocks per sequence in block_table
/// @param seq_lens     [num_seqs]  sequence length for each request
/// @param allocator    BlockAllocator providing K/V data pointers
/// @returns            [num_tokens, num_q_heads, head_dim] attention output
Tensor paged_attention(
    const Tensor& q,
    const int* block_table,
    int max_blocks_per_seq,
    const int* seq_lens,
    class BlockAllocator& allocator);

// ---------------------------------------------------------------------------
// Prefill PagedAttention — single request, P-token chunk
// ---------------------------------------------------------------------------

/// Prefill PagedAttention — reads K/V directly from paged blocks.
///
/// Operates on ONE request's prefill chunk.  Unlike the batched decode
/// paged_attention() which handles N sequences x 1 token, this function
/// handles 1 sequence x P tokens (the prefill chunk).
///
/// The caller MUST first scatter the current chunk's K/V into paged blocks
/// (via scatter_prefill_kv_gpu or the existing scatter_prefill_kv helper)
/// so that the complete KV sequence [0, total_seq) is available in paged
/// memory before calling this function.
///
/// Kernel auto-selection based on P, D, and total sequence length:
///   - D%4 != 0 or <=4 blocks: scalar baseline (can't vectorize)
///   - P <= 64:                direct kernel, Float4 + DoubleBuf
///   - P > 64:                 tiled Flash variant, B_r Q-rows per block
///
/// @param q              Query tensor, shape [P, Hq, D] — the prefill chunk Qs
/// @param block_table    Device pointer to this request's physical block IDs;
///                       size max_blocks (flattened logical-to-physical map).
/// @param max_blocks     Stride of block_table (max blocks per seq).
/// @param seq_len_ptr    Device pointer to a single int: total_seq = historical
///                       KV length + chunk size P.
/// @param allocator      BlockAllocator providing K/V storage raw pointers.
/// @param causal         If true, apply causal mask: query at position
///                       (seq_len - P + qi) attends only to KV positions
///                       <= its own absolute position.  Default false matches
///                       legacy bidirectional-prefill behavior.
///
/// @returns              Attention output, shape [P, Hq, D], F32 on CUDA.
Tensor prefill_paged_attention(
    const Tensor& q,
    const int* block_table,
    int max_blocks,
    const int* seq_len_ptr,
    class BlockAllocator& allocator,
    bool causal = false);

// ---------------------------------------------------------------------------
// GPU Scatter Prefill K/V — write contiguous chunk K/V into paged blocks
// ---------------------------------------------------------------------------

/// Write contiguous K/V tensors into paged blocks — one GPU kernel call.
///
/// Replaces the per-token cudaMemcpy loop in scatter_prefill_kv().
///
/// @param k_contig       [P, Hkv, D] — current chunk's K-projection output
/// @param v_contig       [P, Hkv, D] — current chunk's V-projection output
/// @param start_pos      absolute position in sequence of first chunk token
/// @param n_tokens       number of tokens in this chunk (= P)
/// @param block_table    this sequence's logical-to-physical block map
/// @param max_blocks     stride of block_table
/// @param allocator      BlockAllocator for this layer
void scatter_prefill_kv_gpu(
    const Tensor& k_contig,
    const Tensor& v_contig,
    int start_pos,
    int n_tokens,
    const int* block_table,
    int max_blocks,
    class BlockAllocator& allocator);

// ---------- Batched first-prefill operations -------------------------

/// Batch-scatter first-prefill K/V (historical==0) into paged blocks.
/// One kernel launch for all N entries. One GPU block per token.
void scatter_prefill_kv_batched_gpu(
    const float* k_flat_ptr, const float* v_flat_ptr,
    int Hkv, int D,
    const int* kv_offsets, const int* num_tokens, const int* start_poss,
    const int* token_cumsum, const int* flat_bt, int max_blocks,
    int N, int P_total, class BlockAllocator& allocator);

/// Batched first-prefill attention — contiguous K/V, writes to attn_flat.
/// Reads Q/K/V directly from flat tensors at per-entry offsets.
/// Writes output directly into attn_flat at correct positions.
void first_prefill_attn_batched_gpu(
    float* out_ptr, const float* q_ptr, const float* k_ptr, const float* v_ptr,
    int Hq, int Hkv, int D,
    const int* offsets, const int* kv_offsets, const int* num_tokens,
    const int* token_cumsum, const int* start_poss,
    int N, int P_total, float scale);

}  // namespace kv_cache
}  // namespace lightllm
