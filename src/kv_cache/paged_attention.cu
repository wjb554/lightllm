/// PagedAttention — production-grade decode kernel.
///
/// Optimizations:
///   A. Float4 vectorized loads — K and V loaded as float4 (16-byte) groups
///   B. Double buffering — software pipeline with ping-pong shared-memory tiles
///   C. Split-K — for long sequences (>256 tokens), partition KV blocks across
///      additional grid dimension, then merge partial softmax results
///
/// Kernel selection:
///   - D % 4 != 0  →  baseline (scalar fallback)
///   - seq_len <= SPLIT_K_THRESHOLD → v2 (Float4 + DoubleBuffer)
///   - seq_len >  SPLIT_K_THRESHOLD → splitk + reduce
///
/// fp16 support: all compute kernels are templated on scalar_t.
/// Shared memory always stays float[].  Convert scalar_t→float on load,
/// float→scalar_t on store.  Accumulators (dot products, softmax m/l/acc)
/// stay float throughout.

#include "lightllm/kv_cache/paged_attention.h"
#include "lightllm/kv_cache/block_allocator.h"
#include "lightllm/ops/dispatch.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace lightllm {
namespace kv_cache {

static constexpr int BLOCK              = 16;   // tokens per physical block
static constexpr int WARP               = 32;
static constexpr int MAX_D              = 256;
static constexpr int SPLIT_K            = 4;     // fixed split count
static constexpr int SPLIT_K_THRESHOLD  = 256;   // min seq_len for Split-K
static constexpr int MAX_SHMEM          = 48 * 1024;  // 48 KB per SM

// =========================================================================
// A. Float4 Cooperative Tile Load  (templated on scalar_t)
// =========================================================================
// K/V storage layout: [num_blocks, BLOCK, Hkv, D] row-major.
// Each row has D scalar_t elements.
//
// When scalar_t = float:  float4  = 4 floats.   D must be divisible by 4.
// When scalar_t = half:   float4  = 8 halfs.    D must be divisible by 8.
//
// Shared-memory destination k_tile / v_tile is always float[].

template<typename scalar_t>
__device__ void load_tile_float4(
    float* __restrict__ k_tile,
    float* __restrict__ v_tile,
    const scalar_t* k_pool,
    const scalar_t* v_pool,
    int phys, int kv_head, int tid, int bdim,
    int D, int Hkv)
{
    constexpr int ELEM_PER_F4 = static_cast<int>(sizeof(float4) / sizeof(scalar_t));
    // ELEM_PER_F4 = 4 (float) or 8 (half / bf16)

    const float4* k_pool4 = reinterpret_cast<const float4*>(k_pool);
    const float4* v_pool4 = reinterpret_cast<const float4*>(v_pool);

    int Dgrp   = D / ELEM_PER_F4;               // groups per head-dim
    int ROWgrp = Hkv * Dgrp;                     // groups per (row, kv_head) pair
    int BLKgrp = BLOCK * ROWgrp;                 // groups per physical block
    int Tgrp   = BLOCK * Dgrp;                   // total groups in one tile
    int base   = phys * BLKgrp + kv_head * Dgrp;

    if constexpr (ELEM_PER_F4 == 4) {
        // ---- float path: direct float4 copy (4 floats per group) ----
        float4* k_dst = reinterpret_cast<float4*>(k_tile);
        float4* v_dst = reinterpret_cast<float4*>(v_tile);
        for (int i = tid; i < Tgrp; i += bdim) {
            int row  = i / Dgrp;
            int colg = i - row * Dgrp;
            int idx  = base + row * ROWgrp + colg;
            k_dst[i] = k_pool4[idx];
            v_dst[i] = v_pool4[idx];
        }
    } else {
        // ---- half / bf16 path: unpack 8 elements per float4 load ----
        for (int i = tid; i < Tgrp; i += bdim) {
            int row  = i / Dgrp;
            int colg = i - row * Dgrp;
            int idx  = base + row * ROWgrp + colg;

            float4 k_loaded = k_pool4[idx];
            float4 v_loaded = v_pool4[idx];
            const scalar_t* ks = reinterpret_cast<const scalar_t*>(&k_loaded);
            const scalar_t* vs = reinterpret_cast<const scalar_t*>(&v_loaded);
            int flat_base = row * D + colg * ELEM_PER_F4;

            #pragma unroll
            for (int j = 0; j < ELEM_PER_F4; j++) {
                k_tile[flat_base + j] = static_cast<float>(ks[j]);
                v_tile[flat_base + j] = static_cast<float>(vs[j]);
            }
        }
    }
}

// =========================================================================
// B. Baseline Kernel (scalar loads, single buffer — preserved as fallback)
// =========================================================================
template<typename scalar_t>
__global__ void paged_attention_kernel_baseline(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    int Hq, int Hkv, int D, int max_blocks, float scale)
{
    int token   = blockIdx.x;
    int q_head  = blockIdx.y;
    int kv_head = q_head / (Hq / Hkv);
    int tid     = threadIdx.x;

    extern __shared__ float smem[];
    float* q_smem = smem;                     // [D]
    float* k_tile = smem + D;                 // [BLOCK * D]
    float* v_tile = smem + D + BLOCK * D;     // [BLOCK * D]

    // --- Load full Q into shared memory ---
    for (int d = tid; d < D; d += blockDim.x)
        q_smem[d] = static_cast<float>(q[(token * Hq + q_head) * D + d]);
    __syncthreads();

    // --- Per-thread accumulator (one output dimension) ---
    float acc = 0.0f;
    float m   = -1e38f;
    float l   = 0.0f;

    int seq_len  = seq_lens[token];
    int n_blocks = (seq_len + BLOCK - 1) / BLOCK;

    for (int b = 0; b < n_blocks; b++) {
        int phys = block_table[token * max_blocks + b];

        // Cooperative load K and V tiles into shared memory
        for (int i = tid; i < BLOCK * D; i += blockDim.x) {
            int row = i / D, col = i % D;
            int off = phys * BLOCK * Hkv * D + row * Hkv * D + kv_head * D + col;
            k_tile[i] = static_cast<float>(k_pool[off]);
            v_tile[i] = static_cast<float>(v_pool[off]);
        }
        __syncthreads();

        // Score each row — each thread computes FULL dot product
        for (int row = 0; row < BLOCK; row++) {
            int pos = b * BLOCK + row;
            if (pos >= seq_len) break;

            float dot = 0.0f;
            for (int d = 0; d < D; d++)
                dot += q_smem[d] * k_tile[row * D + d];
            dot *= scale;

            float m_old = m;
            m = fmaxf(m_old, dot);
            float correction = expf(m_old - m);
            float new_term   = expf(dot - m);
            l   = l   * correction + new_term;
            acc = acc * correction + new_term * v_tile[row * D + tid];
        }
        __syncthreads();
    }

    if (tid < D)
        out[(token * Hq + q_head) * D + tid] = static_cast<scalar_t>(acc / (l + 1e-8f));
}

// =========================================================================
// C. V2 Kernel — Float4 + DoubleBuffer (or SingleBuffer for D > 128)
// =========================================================================
// Shared memory layout (double-buffer):  D + 4*BLOCK*D floats
//   q_smem   : [0, D)
//   k_tile0  : [D, D + T)           where T = BLOCK * D
//   v_tile0  : [D+T, D + 2T)
//   k_tile1  : [D+2T, D + 3T)
//   v_tile1  : [D+3T, D + 4T)
//
// For D > 128 the double-buffer would exceed 48 KB shared memory, so the
// kernel falls back to single buffering (k_tile1/v_tile1 alias k_tile0/v_tile0).

template<typename scalar_t>
__global__ void paged_attention_kernel_v2(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    int Hq, int Hkv, int D, int max_blocks, float scale)
{
    int token   = blockIdx.x;
    int q_head  = blockIdx.y;
    int kv_head = q_head / (Hq / Hkv);
    int tid     = threadIdx.x;
    int bdim    = blockDim.x;

    extern __shared__ float smem[];
    float* q_smem  = smem;
    float* k_tile0 = smem + D;
    float* v_tile0 = smem + D + BLOCK * D;

    int T = BLOCK * D;
    // Double buffering requires D + 4*T floats; fall back to single if too large
    bool use_db = (D + 4 * T) * (int)sizeof(float) <= MAX_SHMEM;
    float* k_tile1 = use_db ? (smem + D + 2 * T) : k_tile0;
    float* v_tile1 = use_db ? (smem + D + 3 * T) : v_tile0;

    // --- Load Q into shared memory ---
    for (int d = tid; d < D; d += bdim)
        q_smem[d] = static_cast<float>(q[(token * Hq + q_head) * D + d]);
    __syncthreads();

    float acc = 0.0f;
    float m   = -1e38f;
    float l   = 0.0f;

    int seq_len  = seq_lens[token];
    int n_blocks = (seq_len + BLOCK - 1) / BLOCK;

    if (n_blocks == 0) {
        if (tid < D) out[(token * Hq + q_head) * D + tid] = static_cast<scalar_t>(0.0f);
        return;
    }

    if (use_db) {
        // ===== Double-Buffering Path (D <= 128) =====
        //
        // Pipelining: while computing tile [curr], issue loads for the
        // NEXT tile into the alternate buffer. The single __syncthreads()
        // after load + before compute serialises the handoff; warps that
        // finish the previous compute early can start loading the next
        // tile, hiding global-memory latency.

        // Prologue — prefetch first physical block → tile0
        int phys0 = block_table[token * max_blocks + 0];
        load_tile_float4<scalar_t>(k_tile0, v_tile0, k_pool, v_pool,
                                   phys0, kv_head, tid, bdim, D, Hkv);
        __syncthreads();

        int curr = 0;
        for (int b = 0; b < n_blocks; b++) {
            // Issue load for NEXT block → alternate buffer
            if (b + 1 < n_blocks) {
                int next_phys = block_table[token * max_blocks + b + 1];
                float* k_next = (curr == 0) ? k_tile1 : k_tile0;
                float* v_next = (curr == 0) ? v_tile1 : v_tile0;
                load_tile_float4<scalar_t>(k_next, v_next, k_pool, v_pool,
                                           next_phys, kv_head, tid, bdim, D, Hkv);
            }

            __syncthreads();

            // Compute using CURRENT buffer
            float* k_curr = (curr == 0) ? k_tile0 : k_tile1;
            float* v_curr = (curr == 0) ? v_tile0 : v_tile1;

            for (int row = 0; row < BLOCK; row++) {
                int pos = b * BLOCK + row;
                if (pos >= seq_len) break;

                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += q_smem[d] * k_curr[row * D + d];
                dot *= scale;

                float m_old = m;
                m = fmaxf(m_old, dot);
                float correction = expf(m_old - m);
                float new_term   = expf(dot - m);
                l   = l   * correction + new_term;
                acc = acc * correction + new_term * v_curr[row * D + tid];
            }
            __syncthreads();
            curr = 1 - curr;
        }
    } else {
        // ===== Single-Buffer Float4 Path (D > 128) =====
        for (int b = 0; b < n_blocks; b++) {
            int phys = block_table[token * max_blocks + b];
            load_tile_float4<scalar_t>(k_tile0, v_tile0, k_pool, v_pool,
                                       phys, kv_head, tid, bdim, D, Hkv);
            __syncthreads();

            for (int row = 0; row < BLOCK; row++) {
                int pos = b * BLOCK + row;
                if (pos >= seq_len) break;

                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += q_smem[d] * k_tile0[row * D + d];
                dot *= scale;

                float m_old = m;
                m = fmaxf(m_old, dot);
                float correction = expf(m_old - m);
                float new_term   = expf(dot - m);
                l   = l   * correction + new_term;
                acc = acc * correction + new_term * v_tile0[row * D + tid];
            }
            __syncthreads();
        }
    }

    if (tid < D)
        out[(token * Hq + q_head) * D + tid] = static_cast<scalar_t>(acc / (l + 1e-8f));
}

// =========================================================================
// D. Split-K Kernel — same as V2 but over a subrange of blocks
// =========================================================================
// Grid: (T, Hq, num_splits).  Each block processes ceil(n_blocks/S) KV blocks.
// Output — partial softmax state per (token, head, split):
//   partial[base + 0..D-1] : acc  (unscaled weighted sum)
//   partial[base + D]      : l    (softmax denominator)
//   partial[base + D + 1]  : m    (max logit)
//
// Splits with no work write identity: acc=0, l=0, m=-inf.

template<typename scalar_t>
__global__ void paged_attention_kernel_splitk(
    float* __restrict__ partial,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    int Hq, int Hkv, int D, int max_blocks, float scale, int num_splits)
{
    int token   = blockIdx.x;
    int q_head  = blockIdx.y;
    int split   = blockIdx.z;
    int kv_head = q_head / (Hq / Hkv);
    int tid     = threadIdx.x;
    int bdim    = blockDim.x;

    int seq_len  = seq_lens[token];
    int n_blocks = (seq_len + BLOCK - 1) / BLOCK;

    int blocks_per_split = (n_blocks + num_splits - 1) / num_splits;
    int blk_start = split * blocks_per_split;
    int blk_end   = min(blk_start + blocks_per_split, n_blocks);

    // Write identity for splits with no work
    if (blk_start >= n_blocks) {
        if (tid < D) {
            int base = (token * Hq + q_head) * num_splits * (D + 2)
                     + split * (D + 2);
            partial[base + tid] = 0.0f;
        }
        if (tid == 0) {
            int base = (token * Hq + q_head) * num_splits * (D + 2)
                     + split * (D + 2);
            partial[base + D]     = 0.0f;
            partial[base + D + 1] = -1e38f;
        }
        return;
    }

    // --- Shared memory setup (identical to v2) ---
    extern __shared__ float smem[];
    float* q_smem  = smem;
    float* k_tile0 = smem + D;
    float* v_tile0 = smem + D + BLOCK * D;

    int T = BLOCK * D;
    bool use_db = (D + 4 * T) * (int)sizeof(float) <= MAX_SHMEM;
    float* k_tile1 = use_db ? (smem + D + 2 * T) : k_tile0;
    float* v_tile1 = use_db ? (smem + D + 3 * T) : v_tile0;

    for (int d = tid; d < D; d += bdim)
        q_smem[d] = static_cast<float>(q[(token * Hq + q_head) * D + d]);
    __syncthreads();

    float acc = 0.0f;
    float m   = -1e38f;
    float l   = 0.0f;

    if (use_db) {
        // ---- Double-Buffering Path ----
        int first_phys = block_table[token * max_blocks + blk_start];
        load_tile_float4<scalar_t>(k_tile0, v_tile0, k_pool, v_pool,
                                   first_phys, kv_head, tid, bdim, D, Hkv);
        __syncthreads();

        int curr = 0;
        for (int b = blk_start; b < blk_end; b++) {
            if (b + 1 < blk_end) {
                int next_phys = block_table[token * max_blocks + b + 1];
                float* k_next = (curr == 0) ? k_tile1 : k_tile0;
                float* v_next = (curr == 0) ? v_tile1 : v_tile0;
                load_tile_float4<scalar_t>(k_next, v_next, k_pool, v_pool,
                                           next_phys, kv_head, tid, bdim, D, Hkv);
            }
            __syncthreads();

            float* k_curr = (curr == 0) ? k_tile0 : k_tile1;
            float* v_curr = (curr == 0) ? v_tile0 : v_tile1;

            for (int row = 0; row < BLOCK; row++) {
                int pos = b * BLOCK + row;
                if (pos >= seq_len) break;

                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += q_smem[d] * k_curr[row * D + d];
                dot *= scale;

                float m_old = m;
                m = fmaxf(m_old, dot);
                float correction = expf(m_old - m);
                float new_term   = expf(dot - m);
                l   = l   * correction + new_term;
                acc = acc * correction + new_term * v_curr[row * D + tid];
            }
            __syncthreads();
            curr = 1 - curr;
        }
    } else {
        // ---- Single-Buffer Float4 Path ----
        for (int b = blk_start; b < blk_end; b++) {
            int phys = block_table[token * max_blocks + b];
            load_tile_float4<scalar_t>(k_tile0, v_tile0, k_pool, v_pool,
                                       phys, kv_head, tid, bdim, D, Hkv);
            __syncthreads();

            for (int row = 0; row < BLOCK; row++) {
                int pos = b * BLOCK + row;
                if (pos >= seq_len) break;

                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += q_smem[d] * k_tile0[row * D + d];
                dot *= scale;

                float m_old = m;
                m = fmaxf(m_old, dot);
                float correction = expf(m_old - m);
                float new_term   = expf(dot - m);
                l   = l   * correction + new_term;
                acc = acc * correction + new_term * v_tile0[row * D + tid];
            }
            __syncthreads();
        }
    }

    // ---- Write partial result ----
    int partial_base = (token * Hq + q_head) * num_splits * (D + 2)
                     + split * (D + 2);

    if (tid < D)
        partial[partial_base + tid] = acc;
    if (tid == 0) {
        partial[partial_base + D]     = l;
        partial[partial_base + D + 1] = m;
    }
}

// =========================================================================
// E. Reduction Kernel — Merge SPLIT_K partial results via online softmax
// =========================================================================
// Associative merge of partial states (acc_A, l_A, m_A) and (acc_B, l_B, m_B):
//   m_max = max(m_A, m_B)
//   l     = l_A * exp(m_A - m_max) + l_B * exp(m_B - m_max)
//   acc   = acc_A * exp(m_A - m_max) + acc_B * exp(m_B - m_max)
// Finally:  output = acc / (l + epsilon)
//
// Identity splits (l==0) are skipped safely, avoiding NaN from inf - inf.
//
// partial is always float (stores softmax state).
// output is scalar_t to match the input dtype.

template<typename scalar_t>
__global__ void paged_attention_reduce_kernel(
    scalar_t* __restrict__ out,
    const float* __restrict__ partial,
    int T, int Hq, int D, int num_splits)
{
    int token  = blockIdx.x;
    int q_head = blockIdx.y;
    int tid    = threadIdx.x;

    int out_base     = (token * Hq + q_head) * D;
    int partial_base = (token * Hq + q_head) * num_splits * (D + 2);

    float m   = -1e38f;
    float l   = 0.0f;
    float acc = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        int off_s = s * (D + 2);
        float l_s = partial[partial_base + off_s + D];

        // Identity split — contributes nothing, skip to avoid NaN
        if (l_s == 0.0f) continue;

        float m_s = partial[partial_base + off_s + D + 1];

        if (l == 0.0f) {
            // First non-identity split: initialise state
            m = m_s;
            l = l_s;
            if (tid < D) acc = partial[partial_base + off_s + tid];
            continue;
        }

        float m_max = fmaxf(m, m_s);

        if (m < m_s) {
            // New partial has the larger maximum — scale old accumulator down
            float scale_old = expf(m - m_max);
            l   = l   * scale_old + l_s;
            acc = acc * scale_old;
            if (tid < D)
                acc += partial[partial_base + off_s + tid];
        } else {
            // Existing maximum dominates — scale new partial down
            float scale_new = expf(m_s - m_max);
            l   = l + l_s * scale_new;
            if (tid < D)
                acc += partial[partial_base + off_s + tid] * scale_new;
        }
        m = m_max;
    }

    if (tid < D)
        out[out_base + tid] = static_cast<scalar_t>(
            (l == 0.0f) ? 0.0f : (acc / (l + 1e-8f)));
}

// =========================================================================
// Host launch function — auto-selects kernel based on sequence length
// =========================================================================
Tensor paged_attention(
    const Tensor& q, const int* block_table, int max_blocks,
    const int* seq_lens, BlockAllocator& allocator)
{
    int T = q.size(0), Hq = q.size(1), Hkv = allocator.num_kv_heads(), D = q.size(2);

    if (D > MAX_D)
        throw std::runtime_error("paged_attention: head_dim too large");

    float scale = 1.0f / sqrtf((float)D);
    Tensor out({T, Hq, D}, q.dtype(), Device::CUDA);

    // Block dimension: enough threads to cover D, aligned to warp
    int bdim = ((D + WARP - 1) / WARP) * WARP;
    if (bdim < 32) bdim = 32;

    // ---- Compute actual max sequence length from seq_lens (GPU ptr -> CPU) ----
    int max_actual_seq = 0;
    if (T > 0) {
        std::vector<int> h_seq_lens(T);
        cudaMemcpy(h_seq_lens.data(), seq_lens, T * sizeof(int), cudaMemcpyDeviceToHost);
        for (int t = 0; t < T; t++)
            if (h_seq_lens[t] > max_actual_seq) max_actual_seq = h_seq_lens[t];
    }
    int max_actual_blocks = (max_actual_seq + BLOCK - 1) / BLOCK;

    // ---- Kernel selection based on REAL sequence length ----
    // D not divisible by 4 (float) / 8 (half/bf16) -> must use scalar baseline
    // Short sequences (<=64 tok, <=4 blocks): baseline (no overhead)
    // Medium (65-256 tok): Float4 + optional DoubleBuf
    // Long (>256 tok): Float4 + DoubleBuf + Split-K

    DISPATCH_FLOAT_TYPES(q.dtype(), "paged_attention", {
        // Float4 alignment requirement: 4 for float, 8 for half/bf16
        constexpr int f4_align = static_cast<int>(sizeof(float4) / sizeof(scalar_t));
        bool can_float4 = (D % f4_align == 0) && (max_actual_blocks > 4);

        const scalar_t* k_raw = static_cast<const scalar_t*>(allocator.k_storage_raw());
        const scalar_t* v_raw = static_cast<const scalar_t*>(allocator.v_storage_raw());

        if (!can_float4) {
            // Baseline: simple scalar kernel, minimal launch overhead
            size_t smem = (D + 2 * BLOCK * D) * sizeof(float);
            dim3 grid(T, Hq);
            paged_attention_kernel_baseline<scalar_t><<<grid, bdim, smem>>>(
                out.data<scalar_t>(), q.data<scalar_t>(),
                k_raw, v_raw,
                block_table, seq_lens, Hq, Hkv, D, max_blocks, scale);
        } else {
            bool use_split_k = (max_actual_seq > SPLIT_K_THRESHOLD);
            int num_splits = use_split_k ? SPLIT_K : 1;

            // Shared memory size: double-buffer if D <= 128, else single-buffer
            bool use_db = (D + 4 * BLOCK * D) * (int)sizeof(float) <= MAX_SHMEM;
            size_t smem = use_db ? (D + 4 * BLOCK * D) * sizeof(float)
                                 : (D + 2 * BLOCK * D) * sizeof(float);

            if (num_splits == 1) {
                // Medium path: V2 kernel (Float4 + optional DoubleBuffer)
                dim3 grid(T, Hq);
                paged_attention_kernel_v2<scalar_t><<<grid, bdim, smem>>>(
                    out.data<scalar_t>(), q.data<scalar_t>(),
                    k_raw, v_raw,
                    block_table, seq_lens, Hq, Hkv, D, max_blocks, scale);
            } else {
                // ---- Split-K path ----
                // Intermediate partial buffer: [T, Hq, S, D+2]  (always F32)
                Tensor partial({T, Hq, num_splits, D + 2}, DType::F32, Device::CUDA);

                dim3 split_grid(T, Hq, num_splits);
                paged_attention_kernel_splitk<scalar_t><<<split_grid, bdim, smem>>>(
                    partial.data<float>(), q.data<scalar_t>(),
                    k_raw, v_raw,
                    block_table, seq_lens, Hq, Hkv, D, max_blocks, scale, num_splits);

                // Reduction: one block per (token, head)
                int reduce_bdim = ((D + WARP - 1) / WARP) * WARP;
                if (reduce_bdim < 32) reduce_bdim = 32;

                dim3 reduce_grid(T, Hq);
                paged_attention_reduce_kernel<scalar_t>
                    <<<reduce_grid, reduce_bdim>>>(
                        out.data<scalar_t>(), partial.data<float>(),
                        T, Hq, D, num_splits);
            }
        }
    });

    return std::move(out);
}

// =========================================================================
// F. Prefill Direct Kernel — one block per (chunk_token, q_head)
// =========================================================================
// Grid: dim3(P, Hq)
// Block: bdim threads (warp-aligned, at least D)
// Shared memory: D + 2*BLOCK*D (single) or D + 4*BLOCK*D (double)
//
// Block table is per-request: block_table[b] (not batched like decode).
//
// USE_FLOAT4=true  → vectorized K/V loads via float4
// USE_FLOAT4=false → scalar K/V loads (fallback when D alignment fails or n_blocks<=4)

template<bool USE_FLOAT4, typename scalar_t>
__global__ void prefill_paged_attention_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_len_ptr,
    int P, int Hq, int Hkv, int D, int max_blocks,
    float scale, bool causal)
{
    int chunk_idx = blockIdx.x;  // 0 .. P-1
    int q_head    = blockIdx.y;
    int kv_head   = q_head / (Hq / Hkv);
    int tid       = threadIdx.x;
    int bdim      = blockDim.x;

    int total_seq = *seq_len_ptr;
    int n_blocks  = (total_seq + BLOCK - 1) / BLOCK;
    int abs_q_pos = total_seq - P + chunk_idx;   // absolute position in full sequence

    extern __shared__ float smem[];
    float* q_smem  = smem;
    float* k_tile0 = smem + D;
    float* v_tile0 = smem + D + BLOCK * D;

    int T = BLOCK * D;
    bool use_db = (D + 4 * T) * (int)sizeof(float) <= MAX_SHMEM;
    float* k_tile1 = use_db ? (smem + D + 2 * T) : k_tile0;
    float* v_tile1 = use_db ? (smem + D + 3 * T) : v_tile0;

    // --- Load Q[chunk_idx, q_head, :] into shared memory ---
    for (int d = tid; d < D; d += bdim)
        q_smem[d] = static_cast<float>(q[(chunk_idx * Hq + q_head) * D + d]);
    __syncthreads();

    float acc = 0.0f;
    float m   = -1e38f;
    float l   = 0.0f;

    if (n_blocks == 0) {
        if (tid < D)
            out[(chunk_idx * Hq + q_head) * D + tid] = static_cast<scalar_t>(0.0f);
        return;
    }

    if (USE_FLOAT4) {
        if (use_db) {
            // ---- Double-Buffering Float4 Path (D <= 128) ----
            int phys0 = block_table[0];
            load_tile_float4<scalar_t>(k_tile0, v_tile0, k_pool, v_pool,
                                       phys0, kv_head, tid, bdim, D, Hkv);
            __syncthreads();

            int curr = 0;
            for (int b = 0; b < n_blocks; b++) {
                // Prefetch next block into alternate buffer
                if (b + 1 < n_blocks) {
                    int next_phys = block_table[b + 1];
                    float* k_next = (curr == 0) ? k_tile1 : k_tile0;
                    float* v_next = (curr == 0) ? v_tile1 : v_tile0;
                    load_tile_float4<scalar_t>(k_next, v_next, k_pool, v_pool,
                                               next_phys, kv_head, tid, bdim, D, Hkv);
                }
                __syncthreads();

                float* k_curr = (curr == 0) ? k_tile0 : k_tile1;
                float* v_curr = (curr == 0) ? v_tile0 : v_tile1;

                for (int row = 0; row < BLOCK; row++) {
                    int pos = b * BLOCK + row;
                    if (pos >= total_seq) break;
                    if (causal && pos > abs_q_pos) break;

                    float dot = 0.0f;
                    for (int d = 0; d < D; d++)
                        dot += q_smem[d] * k_curr[row * D + d];
                    dot *= scale;

                    float m_old = m;
                    m = fmaxf(m_old, dot);
                    float correction = expf(m_old - m);
                    float new_term   = expf(dot - m);
                    l   = l   * correction + new_term;
                    acc = acc * correction + new_term * v_curr[row * D + tid];
                }
                __syncthreads();
                curr = 1 - curr;
            }
        } else {
            // ---- Single-Buffer Float4 Path (D > 128) ----
            for (int b = 0; b < n_blocks; b++) {
                int phys = block_table[b];
                load_tile_float4<scalar_t>(k_tile0, v_tile0, k_pool, v_pool,
                                           phys, kv_head, tid, bdim, D, Hkv);
                __syncthreads();

                for (int row = 0; row < BLOCK; row++) {
                    int pos = b * BLOCK + row;
                    if (pos >= total_seq) break;
                    if (causal && pos > abs_q_pos) break;

                    float dot = 0.0f;
                    for (int d = 0; d < D; d++)
                        dot += q_smem[d] * k_tile0[row * D + d];
                    dot *= scale;

                    float m_old = m;
                    m = fmaxf(m_old, dot);
                    float correction = expf(m_old - m);
                    float new_term   = expf(dot - m);
                    l   = l   * correction + new_term;
                    acc = acc * correction + new_term * v_tile0[row * D + tid];
                }
                __syncthreads();
            }
        }
    } else {
        // ---- Scalar Fallback Path (D alignment fails or n_blocks <= 4) ----
        for (int b = 0; b < n_blocks; b++) {
            int phys = block_table[b];

            // Cooperative scalar load of K/V tile
            for (int i = tid; i < BLOCK * D; i += bdim) {
                int row = i / D, col = i % D;
                int off = phys * BLOCK * Hkv * D
                        + row  * Hkv * D
                        + kv_head * D + col;
                k_tile0[i] = static_cast<float>(k_pool[off]);
                v_tile0[i] = static_cast<float>(v_pool[off]);
            }
            __syncthreads();

            for (int row = 0; row < BLOCK; row++) {
                int pos = b * BLOCK + row;
                if (pos >= total_seq) break;
                if (causal && pos > abs_q_pos) break;

                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += q_smem[d] * k_tile0[row * D + d];
                dot *= scale;

                float m_old = m;
                m = fmaxf(m_old, dot);
                float correction = expf(m_old - m);
                float new_term   = expf(dot - m);
                l   = l   * correction + new_term;
                acc = acc * correction + new_term * v_tile0[row * D + tid];
            }
            __syncthreads();
        }
    }

    if (tid < D)
        out[(chunk_idx * Hq + q_head) * D + tid] = static_cast<scalar_t>(acc / (l + 1e-8f));
}

// =========================================================================
// G. Prefill Flash Kernel — tiled, B_r Q-rows per block
// =========================================================================
// Grid: dim3((P + B_r - 1) / B_r, Hq)
// Block: bdim threads (warp-aligned, at least D)
// Shared memory: B_r*D + 2*BLOCK*D (single) or B_r*D + 4*BLOCK*D (double)
//
// Each thread block processes B_r Q rows x one q_head.
// Each thread handles ONE output dimension across all B_r rows.
// K/V tiles are loaded once and shared across B_r Q rows, improving
// arithmetic intensity vs the direct kernel.

template<typename scalar_t>
__global__ void prefill_paged_attention_flash_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_len_ptr,
    int P, int Hq, int Hkv, int D, int max_blocks,
    float scale, bool causal,
    int B_r, int B_c)
{
    int q_head       = blockIdx.y;
    int kv_head      = q_head / (Hq / Hkv);
    int q_tile_start = blockIdx.x * B_r;
    int q_tile_rows  = min(B_r, P - q_tile_start);
    int tid          = threadIdx.x;
    int bdim         = blockDim.x;

    int total_seq = *seq_len_ptr;
    int n_blocks  = (total_seq + B_c - 1) / B_c;

    // Shared memory layout:
    //   q_tile  : [0,                B_r * D)
    //   k_tile0 : [B_r*D,            B_r*D + B_c*D)
    //   v_tile0 : [B_r*D + B_c*D,    B_r*D + 2*B_c*D)
    //   k_tile1 : [B_r*D + 2*B_c*D,  B_r*D + 3*B_c*D)
    //   v_tile1 : [B_r*D + 3*B_c*D,  B_r*D + 4*B_c*D)
    extern __shared__ float smem[];
    float* q_tile  = smem;
    float* k_tile0 = smem + B_r * D;
    float* v_tile0 = smem + B_r * D + B_c * D;

    int T = B_c * D;
    bool use_db = (B_r * D + 4 * T) * (int)sizeof(float) <= MAX_SHMEM;
    float* k_tile1 = use_db ? (smem + B_r * D + 2 * T) : k_tile0;
    float* v_tile1 = use_db ? (smem + B_r * D + 3 * T) : v_tile0;

    // --- Cooperative load B_r Q rows into shared memory ---
    int q_tile_elems = B_r * D;
    for (int i = tid; i < q_tile_elems; i += bdim) {
        int qi = i / D;
        int d  = i % D;
        int global_row = q_tile_start + qi;
        q_tile[i] = (global_row < P)
            ? static_cast<float>(q[(global_row * Hq + q_head) * D + d])
            : 0.0f;
    }
    __syncthreads();

    // --- Per-row online softmax state in registers ---
    // Each thread accumulates ONE output dimension (tid) across B_r rows.
    // B_r is bounded by 64 (MAX_BR) so fixed-size register arrays are safe.
    constexpr int MAX_BR = 64;
    float m_val[MAX_BR];
    float l_val[MAX_BR];
    float acc_val[MAX_BR];

    for (int qi = 0; qi < q_tile_rows; qi++) {
        m_val[qi]   = -1e38f;
        l_val[qi]   = 0.0f;
        acc_val[qi] = 0.0f;
    }

    if (n_blocks == 0) {
        for (int qi = 0; qi < q_tile_rows; qi++) {
            int global_row = q_tile_start + qi;
            if (tid < D)
                out[(global_row * Hq + q_head) * D + tid] = static_cast<scalar_t>(0.0f);
        }
        return;
    }

    if (use_db) {
        // ---- Double-Buffering Path ----
        int phys0 = block_table[0];
        load_tile_float4<scalar_t>(k_tile0, v_tile0, k_pool, v_pool,
                                   phys0, kv_head, tid, bdim, D, Hkv);
        __syncthreads();

        int curr = 0;
        for (int b = 0; b < n_blocks; b++) {
            // Prefetch next block
            if (b + 1 < n_blocks) {
                int next_phys = block_table[b + 1];
                float* k_next = (curr == 0) ? k_tile1 : k_tile0;
                float* v_next = (curr == 0) ? v_tile1 : v_tile0;
                load_tile_float4<scalar_t>(k_next, v_next, k_pool, v_pool,
                                           next_phys, kv_head, tid, bdim, D, Hkv);
            }
            __syncthreads();

            float* k_curr = (curr == 0) ? k_tile0 : k_tile1;
            float* v_curr = (curr == 0) ? v_tile0 : v_tile1;

            // Compute attention for all Q rows inside this tile
            for (int qi = 0; qi < q_tile_rows; qi++) {
                int abs_q_pos = total_seq - P + q_tile_start + qi;

                for (int kj = 0; kj < B_c; kj++) {
                    int pos = b * B_c + kj;
                    if (pos >= total_seq) break;
                    if (causal && pos > abs_q_pos) break;

                    // Full dot product: q_tile[qi][:] dot k_tile[kj][:]
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++)
                        dot += q_tile[qi * D + d] * k_curr[kj * D + d];
                    dot *= scale;

                    // Online softmax update for row qi
                    float m_old = m_val[qi];
                    m_val[qi] = fmaxf(m_old, dot);
                    float correction = expf(m_old - m_val[qi]);
                    float new_term   = expf(dot - m_val[qi]);
                    l_val[qi]   = l_val[qi]   * correction + new_term;
                    acc_val[qi] = acc_val[qi] * correction
                                + new_term * v_curr[kj * D + tid];
                }
            }
            __syncthreads();
            curr = 1 - curr;
        }
    } else {
        // ---- Single-Buffer Path ----
        for (int b = 0; b < n_blocks; b++) {
            int phys = block_table[b];
            load_tile_float4<scalar_t>(k_tile0, v_tile0, k_pool, v_pool,
                                       phys, kv_head, tid, bdim, D, Hkv);
            __syncthreads();

            for (int qi = 0; qi < q_tile_rows; qi++) {
                int abs_q_pos = total_seq - P + q_tile_start + qi;

                for (int kj = 0; kj < B_c; kj++) {
                    int pos = b * B_c + kj;
                    if (pos >= total_seq) break;
                    if (causal && pos > abs_q_pos) break;

                    float dot = 0.0f;
                    for (int d = 0; d < D; d++)
                        dot += q_tile[qi * D + d] * k_tile0[kj * D + d];
                    dot *= scale;

                    float m_old = m_val[qi];
                    m_val[qi] = fmaxf(m_old, dot);
                    float correction = expf(m_old - m_val[qi]);
                    float new_term   = expf(dot - m_val[qi]);
                    l_val[qi]   = l_val[qi]   * correction + new_term;
                    acc_val[qi] = acc_val[qi] * correction
                                + new_term * v_tile0[kj * D + tid];
                }
            }
            __syncthreads();
        }
    }

    // --- Write output: one dimension per thread, B_r output rows ---
    for (int qi = 0; qi < q_tile_rows; qi++) {
        int global_row = q_tile_start + qi;
        if (tid < D)
            out[(global_row * Hq + q_head) * D + tid] =
                static_cast<scalar_t>(acc_val[qi] / (l_val[qi] + 1e-8f));
    }
}

// =========================================================================
// H. Scatter Prefill KV Kernel — write contiguous K/V into paged blocks
// =========================================================================
// Grid: dim3(P) — one block per chunk token
// Block: bdim threads (covers Hkv*D elements)
//
// Eliminates the per-token cudaMemcpy loop: each block copies one token's
// K/V row from the contiguous source into the correct physical block slot.

template<typename scalar_t>
__global__ void scatter_prefill_kv_kernel(
    scalar_t* __restrict__ k_pool,
    scalar_t* __restrict__ v_pool,
    const scalar_t* __restrict__ k_src,
    const scalar_t* __restrict__ v_src,
    const int* __restrict__ block_table,
    int start_pos, int P, int Hkv, int D)
{
    int token   = blockIdx.x;           // 0 .. P-1
    int pos     = start_pos + token;    // absolute position in sequence
    int blk_idx = pos / BLOCK;
    int offset  = pos % BLOCK;
    int phys    = block_table[blk_idx];
    int tid     = threadIdx.x;

    int total    = Hkv * D;
    int src_base = token  * total;
    int dst_base = phys * BLOCK * Hkv * D
                 + offset * Hkv * D;

    for (int i = tid; i < total; i += blockDim.x) {
        k_pool[dst_base + i] = k_src[src_base + i];
        v_pool[dst_base + i] = v_src[src_base + i];
    }
}

// =========================================================================
// I. Host Launch — prefill_paged_attention
// =========================================================================
// Auto-selects between direct kernel (P <= 64) and flash kernel (P > 64),
// with Float4 vectorization when D alignment permits and n_blocks > 4.

Tensor prefill_paged_attention(
    const Tensor& q, const int* block_table, int max_blocks,
    const int* seq_len_ptr, BlockAllocator& allocator, bool causal)
{
    int P = q.size(0), Hq = q.size(1), D = q.size(2);
    int Hkv = allocator.num_kv_heads();

    if (D > MAX_D)
        throw std::runtime_error("prefill_paged_attention: head_dim too large");

    float scale = 1.0f / sqrtf((float)D);
    Tensor out({P, Hq, D}, q.dtype(), Device::CUDA);

    // Block dimension: enough threads to cover D, aligned to warp
    int bdim = ((D + WARP - 1) / WARP) * WARP;
    if (bdim < 32) bdim = 32;

    // ---- Read total_seq from device to decide kernel variant ----
    int total_seq = 0;
    cudaMemcpy(&total_seq, seq_len_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    int n_blocks = (total_seq + BLOCK - 1) / BLOCK;

    bool use_tiled = (P > 64);

    DISPATCH_FLOAT_TYPES(q.dtype(), "prefill_paged_attention", {
        constexpr int f4_align = static_cast<int>(sizeof(float4) / sizeof(scalar_t));
        bool use_float4 = (D % f4_align == 0 && n_blocks > 4);

        const scalar_t* k_raw = static_cast<const scalar_t*>(allocator.k_storage_raw());
        const scalar_t* v_raw = static_cast<const scalar_t*>(allocator.v_storage_raw());

        if (!use_float4) {
            // ---- Scalar baseline ----
            size_t smem = (D + 2 * BLOCK * D) * sizeof(float);
            dim3 grid(P, Hq);
            prefill_paged_attention_kernel<false, scalar_t>
                <<<grid, bdim, smem>>>(
                    out.data<scalar_t>(), q.data<scalar_t>(),
                    k_raw, v_raw, block_table, seq_len_ptr,
                    P, Hq, Hkv, D, max_blocks, scale, causal);
        } else if (!use_tiled) {
            // ---- Direct kernel: Float4 + optional DoubleBuffer ----
            bool use_db = (D + 4 * BLOCK * D) * (int)sizeof(float) <= MAX_SHMEM;
            size_t smem = use_db ? (D + 4 * BLOCK * D) * sizeof(float)
                                 : (D + 2 * BLOCK * D) * sizeof(float);
            dim3 grid(P, Hq);
            prefill_paged_attention_kernel<true, scalar_t>
                <<<grid, bdim, smem>>>(
                    out.data<scalar_t>(), q.data<scalar_t>(),
                    k_raw, v_raw, block_table, seq_len_ptr,
                    P, Hq, Hkv, D, max_blocks, scale, causal);
        } else {
            // ---- Tiled Flash variant (P > 64) ----
            // B_r: Q rows per tile, tuned per head dim to fit shared memory
            // B_c: K/V rows per tile = BLOCK (one physical block = 16 tokens)
            int B_r_eff = (D <= 64) ? 64 : ((D <= 128) ? 48 : 32);
            int B_c     = BLOCK;

            bool use_db = (B_r_eff * D + 4 * B_c * D) * (int)sizeof(float)
                             <= MAX_SHMEM;
            size_t smem = use_db ? (B_r_eff * D + 4 * B_c * D) * sizeof(float)
                                 : (B_r_eff * D + 2 * B_c * D) * sizeof(float);

            dim3 grid((P + B_r_eff - 1) / B_r_eff, Hq);
            prefill_paged_attention_flash_kernel<scalar_t>
                <<<grid, bdim, smem>>>(
                    out.data<scalar_t>(), q.data<scalar_t>(),
                    k_raw, v_raw, block_table, seq_len_ptr,
                    P, Hq, Hkv, D, max_blocks, scale, causal,
                    B_r_eff, B_c);
        }
    });

    return std::move(out);
}

// =========================================================================
// J. Host Launch — scatter_prefill_kv_gpu
// =========================================================================
void scatter_prefill_kv_gpu(
    const Tensor& k_contig, const Tensor& v_contig,
    int start_pos, int n_tokens,
    const int* block_table, int max_blocks,
    BlockAllocator& allocator)
{
    int Hkv = allocator.num_kv_heads();
    int D   = allocator.head_dim();

    // Cover Hkv*D elements; cap at 256 for reasonable occupancy
    int bdim = (Hkv * D < 256) ? Hkv * D : 256;
    if (bdim < 32) bdim = 32;

    dim3 grid(n_tokens);

    DISPATCH_FLOAT_TYPES(k_contig.dtype(), "scatter_prefill_kv", {
        scalar_t* k_pool = const_cast<scalar_t*>(
            static_cast<const scalar_t*>(allocator.k_storage_raw()));
        scalar_t* v_pool = const_cast<scalar_t*>(
            static_cast<const scalar_t*>(allocator.v_storage_raw()));

        scatter_prefill_kv_kernel<scalar_t><<<grid, bdim>>>(
            k_pool, v_pool,
            k_contig.data<scalar_t>(), v_contig.data<scalar_t>(),
            block_table,
            start_pos, n_tokens, Hkv, D);
    });
}

// =========================================================================
// K. Batched First-Prefill Scatter Kernel
// =========================================================================
// Grid: dim3(P_total) — one block per token across all entries.
// Searches token_cumsum to find entry index, computes physical destination,
// copies one token's K/V row.

template<typename scalar_t>
__global__ void scatter_prefill_kv_batched_kernel(
    scalar_t* __restrict__ k_pool, scalar_t* __restrict__ v_pool,
    const scalar_t* __restrict__ k_src, const scalar_t* __restrict__ v_src,
    int Hkv, int D,
    const int* __restrict__ kv_offsets, const int* __restrict__ num_tokens,
    const int* __restrict__ start_poss, const int* __restrict__ token_cumsum,
    const int* __restrict__ flat_bt, int max_blocks, int N)
{
    int global_token = blockIdx.x;
    int tid = threadIdx.x;

    // Linear scan to find entry (N is small, typically <= 32)
    int entry_idx = 0;
    while (entry_idx < N && global_token >= token_cumsum[entry_idx + 1])
        entry_idx++;

    int local_token = global_token - token_cumsum[entry_idx];
    int pos        = start_poss[entry_idx] + local_token;
    int blk_idx    = pos / BLOCK;
    int offset     = pos % BLOCK;
    int phys       = flat_bt[entry_idx * max_blocks + blk_idx];

    int total    = Hkv * D;
    int src_base = kv_offsets[entry_idx] + local_token * total;
    int dst_base = phys * BLOCK * Hkv * D + offset * Hkv * D;

    for (int i = tid; i < total; i += blockDim.x) {
        k_pool[dst_base + i] = k_src[src_base + i];
        v_pool[dst_base + i] = v_src[src_base + i];
    }
}

void scatter_prefill_kv_batched_gpu(
    const float* k_flat_ptr, const float* v_flat_ptr,
    int Hkv, int D,
    const int* kv_offsets, const int* num_tokens, const int* start_poss,
    const int* token_cumsum, const int* flat_bt, int max_blocks,
    int N, int P_total, BlockAllocator& allocator)
{
    int bdim = (Hkv * D < 256) ? Hkv * D : 256;
    if (bdim < 32) bdim = 32;

    float* k_pool = const_cast<float*>(
        static_cast<const float*>(allocator.k_storage_raw()));
    float* v_pool = const_cast<float*>(
        static_cast<const float*>(allocator.v_storage_raw()));

    scatter_prefill_kv_batched_kernel<float><<<P_total, bdim>>>(
        k_pool, v_pool,
        k_flat_ptr, v_flat_ptr,
        Hkv, D,
        kv_offsets, num_tokens, start_poss,
        token_cumsum, flat_bt,
        max_blocks, N);
    cudaDeviceSynchronize();
}

// fp16-aware overload — dispatches on DType parameter
void scatter_prefill_kv_batched_gpu_dispatch(
    DType dtype,
    const void* k_flat_ptr, const void* v_flat_ptr,
    int Hkv, int D,
    const int* kv_offsets, const int* num_tokens, const int* start_poss,
    const int* token_cumsum, const int* flat_bt, int max_blocks,
    int N, int P_total, BlockAllocator& allocator)
{
    int bdim = (Hkv * D < 256) ? Hkv * D : 256;
    if (bdim < 32) bdim = 32;

    DISPATCH_FLOAT_TYPES(dtype, "scatter_prefill_kv_batched", {
        scalar_t* k_pool = const_cast<scalar_t*>(
            static_cast<const scalar_t*>(allocator.k_storage_raw()));
        scalar_t* v_pool = const_cast<scalar_t*>(
            static_cast<const scalar_t*>(allocator.v_storage_raw()));

        scatter_prefill_kv_batched_kernel<scalar_t><<<P_total, bdim>>>(
            k_pool, v_pool,
            static_cast<const scalar_t*>(k_flat_ptr),
            static_cast<const scalar_t*>(v_flat_ptr),
            Hkv, D,
            kv_offsets, num_tokens, start_poss,
            token_cumsum, flat_bt,
            max_blocks, N);
    });
    cudaDeviceSynchronize();
}

// =========================================================================
// L. Batched First-Prefill Attention — contiguous K/V, direct output
// =========================================================================
// Grid: dim3((P_total + B_r - 1) / B_r, Hq)
// Each block processes B_r Q rows. K/V read from contiguous linear memory
// at per-entry offsets. Output written directly to attn_flat at correct
// entry positions (no output copy-back needed).

template<typename scalar_t>
__global__ void first_prefill_attn_batched_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k, const scalar_t* __restrict__ v,
    int Hq, int Hkv, int D,
    const int* __restrict__ offsets, const int* __restrict__ kv_offsets,
    const int* __restrict__ num_tokens, const int* __restrict__ token_cumsum,
    const int* __restrict__ start_poss,
    int N, int P_total, float scale, int B_r)
{
    int q_head       = blockIdx.y;
    int kv_head      = q_head / (Hq / Hkv);
    int q_tile_start = blockIdx.x * B_r;
    int tid          = threadIdx.x;
    int bdim         = blockDim.x;

    // Round up B_r to multiple of bdim for clean cooperative loads
    int rounded_B_r = ((B_r + bdim - 1) / bdim) * bdim;

    extern __shared__ float smem[];
    float* q_tile = smem;  // [rounded_B_r * D]

    // Cooperative load Q tile from flat q_flat at correct entry offsets
    for (int i = tid; i < rounded_B_r * D; i += bdim) {
        int qi = i / D, d = i % D;
        int global_row = q_tile_start + qi;
        if (global_row < P_total) {
            int entry = 0;
            while (entry < N && global_row >= token_cumsum[entry + 1])
                entry++;
            int local_row = global_row - token_cumsum[entry];
            q_tile[i] = static_cast<float>(
                q[offsets[entry] + local_row * Hq * D + q_head * D + d]);
        } else {
            q_tile[i] = 0.0f;
        }
    }
    __syncthreads();

    constexpr int MAX_BR = 64;
    float m_val[MAX_BR], l_val[MAX_BR], acc_val[MAX_BR];
    int actual_rows = min(B_r, P_total - q_tile_start);

    for (int qi = 0; qi < actual_rows; qi++) {
        m_val[qi] = -1e38f;
        l_val[qi] = 0.0f;
        acc_val[qi] = 0.0f;
    }

    // Process each Q row independently within its own entry
    for (int qi = 0; qi < actual_rows; qi++) {
        int global_row = q_tile_start + qi;
        int entry = 0;
        while (entry < N && global_row >= token_cumsum[entry + 1])
            entry++;
        int local_row  = global_row - token_cumsum[entry];
        int abs_q_pos  = start_poss[entry] + local_row;
        int kv_len     = num_tokens[entry];
        int kv_base    = kv_offsets[entry];

        for (int pos = 0; pos < kv_len; pos++) {
            int abs_pos = start_poss[entry] + pos;
            if (abs_pos > abs_q_pos) break;

            // Dot product: q_tile[qi*D..] dot k_entry[pos*kv_head*D..]
            float dot = 0.0f;
            for (int d = 0; d < D; d++)
                dot += q_tile[qi * D + d]
                     * static_cast<float>(
                         k[kv_base + pos * Hkv * D + kv_head * D + d]);
            dot *= scale;

            // Online softmax update
            float m_old = m_val[qi];
            m_val[qi] = fmaxf(m_old, dot);
            float correction = expf(m_old - m_val[qi]);
            float new_term   = expf(dot - m_val[qi]);
            l_val[qi]   = l_val[qi] * correction + new_term;
            acc_val[qi] = acc_val[qi] * correction
                        + new_term * static_cast<float>(
                            v[kv_base + pos * Hkv * D + kv_head * D + tid]);
        }
    }

    // Write output to attn_flat at correct per-entry positions
    for (int qi = 0; qi < actual_rows; qi++) {
        int global_row = q_tile_start + qi;
        int entry = 0;
        while (entry < N && global_row >= token_cumsum[entry + 1])
            entry++;
        int local_row = global_row - token_cumsum[entry];
        int out_row  = (offsets[entry] / (Hq * D)) + local_row;
        if (tid < D)
            out[(out_row * Hq + q_head) * D + tid] =
                static_cast<scalar_t>(acc_val[qi] / (l_val[qi] + 1e-8f));
    }
}

void first_prefill_attn_batched_gpu(
    float* out_ptr, const float* q_ptr, const float* k_ptr, const float* v_ptr,
    int Hq, int Hkv, int D,
    const int* offsets, const int* kv_offsets, const int* num_tokens,
    const int* token_cumsum, const int* start_poss,
    int N, int P_total, float scale)
{
    int bdim = ((D + WARP - 1) / WARP) * WARP;
    if (bdim < 32) bdim = 32;

    // B_r: Q rows per tile, tuned per head dim
    int B_r = (D <= 64) ? 64 : ((D <= 128) ? 48 : 32);
    int rounded_B_r = ((B_r + bdim - 1) / bdim) * bdim;

    size_t smem = rounded_B_r * D * sizeof(float);

    dim3 grid((P_total + B_r - 1) / B_r, Hq);
    first_prefill_attn_batched_kernel<float><<<grid, bdim, smem>>>(
        out_ptr, q_ptr, k_ptr, v_ptr,
        Hq, Hkv, D,
        offsets, kv_offsets, num_tokens, token_cumsum, start_poss,
        N, P_total, scale, B_r);
    cudaDeviceSynchronize();
}

// fp16-aware overload — dispatches on DType parameter
void first_prefill_attn_batched_gpu_dispatch(
    DType dtype,
    void* out_ptr, const void* q_ptr, const void* k_ptr, const void* v_ptr,
    int Hq, int Hkv, int D,
    const int* offsets, const int* kv_offsets, const int* num_tokens,
    const int* token_cumsum, const int* start_poss,
    int N, int P_total, float scale)
{
    int bdim = ((D + WARP - 1) / WARP) * WARP;
    if (bdim < 32) bdim = 32;

    // B_r: Q rows per tile, tuned per head dim
    int B_r = (D <= 64) ? 64 : ((D <= 128) ? 48 : 32);
    int rounded_B_r = ((B_r + bdim - 1) / bdim) * bdim;

    size_t smem = rounded_B_r * D * sizeof(float);

    dim3 grid((P_total + B_r - 1) / B_r, Hq);

    DISPATCH_FLOAT_TYPES(dtype, "first_prefill_attn_batched", {
        first_prefill_attn_batched_kernel<scalar_t><<<grid, bdim, smem>>>(
            static_cast<scalar_t*>(out_ptr),
            static_cast<const scalar_t*>(q_ptr),
            static_cast<const scalar_t*>(k_ptr),
            static_cast<const scalar_t*>(v_ptr),
            Hq, Hkv, D,
            offsets, kv_offsets, num_tokens, token_cumsum, start_poss,
            N, P_total, scale, B_r);
    });
    cudaDeviceSynchronize();
}

}  // namespace kv_cache
}  // namespace lightllm
