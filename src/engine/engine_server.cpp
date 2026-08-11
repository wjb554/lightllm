/// LightLLM Engine Server — heterogeneous batch execution with PagedAttention.
///
/// INTEGRATION POINT with Scheduler:
///   The Scheduler produces a ScheduleStep (heterogeneous batch of prefill chunks
///   and decode tokens).  The EngineServer consumes it via step(), which runs the
///   full transformer forward pass for ALL tokens in the batch, then returns
///   sampled tokens for decode entries.  Prefill entries update internal KV cache
///   state but produce no output token from that call.
///
/// LIFECYCLE:
///   1. main loop calls engine.create_request_state(req) when a new request arrives
///   2. main loop calls scheduler.add_request(req)
///   3. scheduler.step() -> ScheduleStep
///   4. engine.step(schedule_step, states) -> vector<SampledToken>
///   5. main loop feeds tokens back, marks finished requests
///   6. engine.release_request(state) when request is done
///
/// BLOCK MANAGEMENT:
///   - Prefill first chunk:  allocate blocks, scatter contiguous K/V into them
///   - Prefill later chunks: allocate more blocks, scatter + use paged K/V for history
///   - Decode:               allocate block on boundary crossing, write single-token K/V
///   - Finish:               release ALL blocks across ALL layers

#include "lightllm/engine/engine.h"
#include "lightllm/engine/scheduler.h"
#include "lightllm/kv_cache/block_allocator.h"
#include "lightllm/kv_cache/paged_attention.h"
#include "lightllm/model/model_config.h"
#include "lightllm/model/model_loader.h"
#include "lightllm/tensor.h"
#include "lightllm/ops/norm.h"
#include "lightllm/ops/rope.h"
#include "lightllm/ops/gemm.h"
#include "lightllm/ops/elementwise.h"
#include "lightllm/ops/sampling.h"
#include "lightllm/ops/attention.h"
#include "lightllm/ops/lm_head.h"
#include "lightllm/ops/embedding.h"
#include "lightllm/ops/argmax.h"
#include "lightllm/tokenizer/tokenizer.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <xgrammar/xgrammar.h>
#include <dlpack/dlpack.h>

namespace lightllm {
namespace engine {

using namespace model;
using namespace ops;

// ============================================================================
// Grammar state rollback for speculative decoding with structured output
// ============================================================================

/// Save grammar state before speculative draft, restore if rejected.
///
/// When speculative decoding is active WITH structured output, draft tokens
/// are generated without grammar constraint (greedy, fast).  Target
/// verification checks grammar compatibility via AcceptToken().  If draft
/// tokens are rejected, this helper rolls the grammar FSM back by
/// `num_to_rollback` steps.
///
/// GrammarMatcher::Rollback() (xgrammar v0.2.5):
///   void Rollback(int num_tokens);
///   // Rolls back the grammar FSM by num_tokens steps.
///   // - Cannot exceed the current number of steps.
///   // - Cannot exceed max_rollback_tokens if set at construction (default: -1 = unlimited).
///
/// Note: AcceptToken() handles a single-token rollback internally when it
/// returns false, so this helper is primarily for bulk rollback when multiple
/// provisionally-accepted tokens must be undone after verification mismatch.
static void rollback_grammar_if_needed(
    xgrammar::GrammarMatcher* matcher,
    int num_to_rollback)
{
    if (matcher && num_to_rollback > 0) {
        matcher->Rollback(num_to_rollback);
    }
}

// ============================================================================
// GPU Prefill Attention Kernels  (copied from engine_paged.cpp)
// ============================================================================

// Stage 1: Q*K^T — one block per (token, Q-head)
template<typename scalar_t>
__global__ void es_attn_qk_kernel(
    float* scores, const scalar_t* q, const scalar_t* k,
    int TQ, int Hq, int TK, int Hkv, int D, int groups, float scale)
{
    int t=blockIdx.x, h=blockIdx.y, kvh=h/groups, tid=threadIdx.x;
    for (int s=tid; s<TK; s+=blockDim.x) {
        float dot=0;
        for (int d=0; d<D; d++) dot+=(float)q[(t*Hq+h)*D+d]*(float)k[(s*Hkv+kvh)*D+d];
        scores[(t*Hq+h)*TK+s]=dot*scale;
    }
}

// Stage 2: Softmax along last dim
__global__ void es_attn_softmax_kernel(float* scores, int rows, int cols) {
    int row=blockIdx.x, tid=threadIdx.x;
    __shared__ float smem[64];
    int wid=tid/32, lid=tid%32, nw=blockDim.x/32;
    float mx=-1e38f;
    for (int i=tid; i<cols; i+=blockDim.x) mx=fmaxf(mx, scores[row*cols+i]);
    for (int o=16; o>0; o>>=1) mx=fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    if (lid==0) smem[wid]=mx; __syncthreads();
    if (tid<32) { float v=(tid<nw)?smem[tid]:-1e38f;
        for (int o=16; o>0; o>>=1) v=fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
        if (tid==0) smem[0]=v; } __syncthreads();
    mx=smem[0]; float sm=0;
    for (int i=tid; i<cols; i+=blockDim.x) {
        float v=expf(scores[row*cols+i]-mx); scores[row*cols+i]=v; sm+=v;
    }
    for (int o=16; o>0; o>>=1) sm+=__shfl_down_sync(0xffffffff, sm, o);
    if (lid==0) smem[wid]=sm; __syncthreads();
    if (tid<32) { float v=0;
        for (int w=0; w<nw; w++) v+=smem[w];
        for (int o=16; o>0; o>>=1) v+=__shfl_down_sync(0xffffffff, v, o);
        if (tid==0) smem[1]=v; } __syncthreads();
    sm=smem[1]; float inv=1.f/(sm+1e-8f);
    for (int i=tid; i<cols; i+=blockDim.x) scores[row*cols+i]*=inv;
}

// Stage 3: S@V — one block per (token, Q-head)
template<typename scalar_t>
__global__ void es_attn_sv_kernel(
    scalar_t* out, const float* scores, const scalar_t* v,
    int TQ, int Hq, int TK, int Hkv, int D, int groups)
{
    int t=blockIdx.x, h=blockIdx.y, kvh=h/groups, tid=threadIdx.x;
    for (int d=tid; d<D; d+=blockDim.x) {
        float acc=0;
        for (int s=0; s<TK; s++)
            acc+=scores[(t*Hq+h)*TK+s]*(float)v[(s*Hkv+kvh)*D+d];
        out[(t*Hq+h)*D+d]=(scalar_t)acc;
    }
}

// FlashAttention-style tiled prefill kernel
static constexpr int B_r=64, B_c=64;

template<typename scalar_t>
__global__ void es_flash_prefill_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k, const scalar_t* __restrict__ v,
    int P, int Hq, int Hkv, int D, int groups, float scale)
{
    int q_head   = blockIdx.y;
    int kv_head  = q_head / groups;
    int q_start  = blockIdx.x * B_r;
    int tid      = threadIdx.x;
    int dim      = tid;

    extern __shared__ float smem[];
    float* q_tile = smem;
    float* k_tile = smem + B_r * D;
    float* v_tile = smem + B_r * D + B_c * D;

    float m_val[B_r], l_val[B_r];
    for (int i = 0; i < B_r; i++) { m_val[i] = -1e38f; l_val[i] = 0.0f; }
    float acc[B_r] = {};

    for (int i = tid; i < B_r * D; i += blockDim.x) {
        int qi = q_start + i / D, d = i % D;
        q_tile[i] = (qi < P) ? (float)q[(qi * Hq + q_head) * D + d] : 0.0f;
    }
    __syncthreads();

    int q_tokens = min(B_r, P - q_start);
    int n_kv_blocks = (P + B_c - 1) / B_c;

    for (int kb = 0; kb < n_kv_blocks; kb++) {
        int kv_start = kb * B_c;
        int kv_tokens = min(B_c, P - kv_start);

        for (int i = tid; i < B_c * D; i += blockDim.x) {
            int ki = kv_start + i / D, d = i % D;
            bool valid = (ki < P);
            k_tile[i] = valid ? (float)k[(ki * Hkv + kv_head) * D + d] : 0.0f;
            v_tile[i] = valid ? (float)v[(ki * Hkv + kv_head) * D + d] : 0.0f;
        }
        __syncthreads();

        for (int qi = 0; qi < q_tokens; qi++) {
            for (int kj = 0; kj < kv_tokens; kj++) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += q_tile[qi * D + d] * k_tile[kj * D + d];
                dot *= scale;

                float m_old = m_val[qi];
                m_val[qi] = fmaxf(m_old, dot);
                float correction = expf(m_old - m_val[qi]);
                float new_term   = expf(dot - m_val[qi]);
                l_val[qi] = l_val[qi] * correction + new_term;
                if (dim < D)
                    acc[qi] = acc[qi] * correction + new_term * v_tile[kj * D + dim];
            }
        }
        __syncthreads();
    }

    if (dim < D) {
        for (int qi = 0; qi < q_tokens; qi++)
            out[((q_start + qi) * Hq + q_head) * D + dim]
                = (scalar_t)(acc[qi] / (l_val[qi] + 1e-8f));
    }
}

// Adaptive prefill attention: short prompts -> 3-stage, long prompts -> tiled.
// q/k/v may be F32 or F16; scores is always F32; output matches q's dtype.
static Tensor prefill_attn(const Tensor& q, const Tensor& k, const Tensor& v,
    int P, int Hq, int Hkv, int D, int groups, float scale)
{
    size_t scores_bytes = (size_t)P * Hq * P * sizeof(float);
    bool use_tiled = (scores_bytes > 32ULL * 1024 * 1024);
    DType dtype = q.dtype();

    if (!use_tiled) {
        // Scheme A: 3-stage
        Tensor scores({P * Hq, P}, DType::F32, Device::CUDA);
        dim3 g1(P, Hq), b1(std::min(256, D));
        dim3 g3(P, Hq);
        int rows=P*Hq, blk=std::min(512, ((P+31)/32)*32);
        if (blk<32) blk=32;

        if (dtype == DType::F16) {
            Tensor out({P, Hq, D}, DType::F16, Device::CUDA);
            es_attn_qk_kernel<half><<<g1,b1>>>(scores.data<float>(),
                q.data<half>(), k.data<half>(),
                P, Hq, P, Hkv, D, groups, scale);
            es_attn_softmax_kernel<<<rows,blk>>>(scores.data<float>(), rows, P);
            es_attn_sv_kernel<half><<<g3,b1>>>(out.data<half>(),
                scores.data<float>(), v.data<half>(),
                P, Hq, P, Hkv, D, groups);
            return out;
        } else {
            Tensor out({P, Hq, D}, DType::F32, Device::CUDA);
            es_attn_qk_kernel<float><<<g1,b1>>>(scores.data<float>(),
                q.data<float>(), k.data<float>(),
                P, Hq, P, Hkv, D, groups, scale);
            es_attn_softmax_kernel<<<rows,blk>>>(scores.data<float>(), rows, P);
            es_attn_sv_kernel<float><<<g3,b1>>>(out.data<float>(),
                scores.data<float>(), v.data<float>(),
                P, Hq, P, Hkv, D, groups);
            return out;
        }
    } else {
        // Scheme B: FlashAttention tiled
        int bdim = ((D + 31) / 32) * 32; if (bdim < 32) bdim = 32;
        dim3 grid((P + B_r - 1) / B_r, Hq);

        if (dtype == DType::F16) {
            size_t smem = (B_r * D + 2 * B_c * D) * sizeof(float);
            Tensor out({P, Hq, D}, DType::F16, Device::CUDA);
            es_flash_prefill_kernel<half><<<grid, bdim, smem>>>(
                out.data<half>(), q.data<half>(), k.data<half>(), v.data<half>(),
                P, Hq, Hkv, D, groups, scale);
            return out;
        } else {
            size_t smem = (B_r * D + 2 * B_c * D) * sizeof(float);
            Tensor out({P, Hq, D}, DType::F32, Device::CUDA);
            es_flash_prefill_kernel<float><<<grid, bdim, smem>>>(
                out.data<float>(), q.data<float>(), k.data<float>(), v.data<float>(),
                P, Hq, Hkv, D, groups, scale);
            return out;
        }
    }
}

// ============================================================================
// Weight loading helpers
// ============================================================================

static std::vector<float> bf16f32(const std::vector<char>& r) {
    std::vector<float> f(r.size()/2);
    for (size_t i=0; i<f.size(); i++) {
        uint16_t b=((const uint16_t*)r.data())[i];
        uint32_t u=b<<16;
        f[i]=*(float*)&u;
    }
    return f;
}

static Tensor load_f32(SafetensorsLoader& l, const std::string& n) {
    auto* info = l.get_info(n);
    Tensor cpu = l.load_tensor(n);
    std::vector<char> raw(cpu.nbytes());
    cpu.copy_to(raw.data(), cpu.nbytes());
    auto f32 = bf16f32(raw);
    Tensor gpu(info->shape, DType::F32, Device::CUDA);
    gpu.copy_from(f32.data(), f32.size()*sizeof(float));
    return gpu;
}

/// Load a weight tensor as F16 on GPU.
/// Reads the raw bytes (assumed BF16, 2 bytes/element), converts
/// BF16→float→half on the CPU, then uploads to a GPU F16 tensor.
static Tensor load_f16(SafetensorsLoader& l, const std::string& n) {
    auto* info = l.get_info(n);
    Tensor cpu = l.load_tensor(n);
    std::vector<char> raw(cpu.nbytes());
    cpu.copy_to(raw.data(), cpu.nbytes());
    size_t nel = raw.size() / 2;
    std::vector<half> f16(nel);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
    for (size_t i = 0; i < nel; i++) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        f16[i] = __float2half(*reinterpret_cast<float*>(&bits));
    }
    Tensor gpu(info->shape, DType::F16, Device::CUDA);
    gpu.copy_from(f16.data(), f16.size() * sizeof(half));
    return gpu;
}

/// Dispatcher: loads a weight as F16 if use_fp16 is true, otherwise F32.
static Tensor load_weight(SafetensorsLoader& l, const std::string& n,
                          bool use_fp16) {
    return use_fp16 ? load_f16(l, n) : load_f32(l, n);
}

// ============================================================================
// DType conversion helpers (F32 <-> F16)
// ============================================================================

/// Simple element-wise F32 -> F16 conversion kernel.
/// Used at the attention boundary: attention outputs F32, but the
/// residual path (O-projection, add) expects weight_dtype_ (F16).
__global__ void cast_f32_to_f16_kernel(half* __restrict__ dst,
                                        const float* __restrict__ src,
                                        int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __float2half(src[i]);
}

static Tensor cast_f32_to_f16(const Tensor& x) {
    int N = static_cast<int>(x.numel());
    Tensor out(x.shape(), DType::F16, Device::CUDA);
    int block = 256, grid = (N + 255) / 256;
    cast_f32_to_f16_kernel<<<grid, block>>>(out.data<half>(), x.data<float>(), N);
    return out;
}

/// Simple element-wise F16 -> F32 conversion kernel (unused currently,
/// but available for completeness if we ever need to go the other way).
__global__ void cast_f16_to_f32_kernel(float* __restrict__ dst,
                                        const half* __restrict__ src,
                                        int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __half2float(src[i]);
}

static Tensor cast_f16_to_f32(const Tensor& x) {
    int N = static_cast<int>(x.numel());
    Tensor out(x.shape(), DType::F32, Device::CUDA);
    int block = 256, grid = (N + 255) / 256;
    cast_f16_to_f32_kernel<<<grid, block>>>(out.data<float>(), x.data<half>(), N);
    return out;
}

// ============================================================================
// EngineServer method implementations (class declared in engine.h)
// ============================================================================

// ============================================================================
// Constructor — load model, allocate KV pools, pre-allocate batch scratch
// ============================================================================

EngineServer::EngineServer(const std::string& model_dir,
                           int max_seq_len,
                           int max_batch_tokens,
                           int kv_cache_mb,
                           kv_cache::PrefixCachePolicy prefix_cache_policy,
                           bool use_fp16,
                           bool budget_from_free)
    : max_batch_tokens_(max_batch_tokens)
    , kv_cache_mb_(kv_cache_mb)
    , use_fp16_(use_fp16)
    , budget_from_free_(budget_from_free)
    , weight_dtype_(use_fp16 ? DType::F16 : DType::F32)
    , prefix_cache_policy_(prefix_cache_policy)
{
    char buf[512];
    snprintf(buf, sizeof(buf), "%s/config.json", model_dir.c_str());
    cfg_ = parse_config(buf);
    snprintf(buf, sizeof(buf), "%s/model.safetensors", model_dir.c_str());
    SafetensorsLoader loader(buf);

    D_   = cfg_.hidden_size;
    Hq_  = cfg_.num_attention_heads;
    Hkv_ = cfg_.num_key_value_heads;
    hd_  = cfg_.head_dim;
    n_layers_ = cfg_.num_hidden_layers;
    vocab_    = cfg_.vocab_size;

    if (max_seq_len <= 0)
        max_seq_len_ = cfg_.max_position_embeddings;
    else
        max_seq_len_ = max_seq_len;

    // kv_cache_mb=0: derive from max_seq_len (legacy). >0: use exact budget.
    if (kv_cache_mb_ > 0) {
        // Per-layer budget in bytes
        size_t per_layer_bytes = static_cast<size_t>(kv_cache_mb_) * 1048576 / n_layers_;
        size_t block_bytes = static_cast<size_t>(BLOCK_SIZE) * Hkv_ * hd_ * 2 * sizeof(float);
        num_blocks_ = static_cast<int>(per_layer_bytes / block_bytes);
        if (num_blocks_ < 1) num_blocks_ = 1;
    } else {
        num_blocks_ = (max_seq_len_ + BLOCK_SIZE - 1) / BLOCK_SIZE + 8;
    }
    max_blocks_per_seq_ = (max_seq_len_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("EngineServer: %s, %d layers, max_seq_len=%d, %d blocks "
           "(block_size=%d, max_batch_tokens=%d)\n",
           cfg_.architecture.c_str(), n_layers_,
           max_seq_len_, num_blocks_, BLOCK_SIZE, max_batch_tokens_);
    printf("  prefix_cache=%s\n", kv_cache::prefix_cache_policy_name(prefix_cache_policy_));
    printf("Loading weights...\n");

    embed_w_    = load_weight(loader, "model.embed_tokens.weight", use_fp16);
    final_norm_ = load_weight(loader, "model.norm.weight", use_fp16);
    for (int i = 0; i < n_layers_; i++) {
        auto ns = std::to_string(i);
        layers_.push_back(std::make_unique<EngineLayerW>(EngineLayerW{
            load_weight(loader, "model.layers."+ns+".self_attn.q_proj.weight", use_fp16),
            load_weight(loader, "model.layers."+ns+".self_attn.k_proj.weight", use_fp16),
            load_weight(loader, "model.layers."+ns+".self_attn.v_proj.weight", use_fp16),
            load_weight(loader, "model.layers."+ns+".self_attn.o_proj.weight", use_fp16),
            load_weight(loader, "model.layers."+ns+".input_layernorm.weight", use_fp16),
            load_weight(loader, "model.layers."+ns+".mlp.gate_proj.weight", use_fp16),
            load_weight(loader, "model.layers."+ns+".mlp.up_proj.weight", use_fp16),
            load_weight(loader, "model.layers."+ns+".mlp.down_proj.weight", use_fp16),
            load_weight(loader, "model.layers."+ns+".post_attention_layernorm.weight", use_fp16),
        }));
        if (i == 0 || i == n_layers_ - 1)
            printf("  layer %d loaded\n", i);
    }
    printf("All weights loaded (%s on GPU)\n", use_fp16 ? "FP16" : "FP32");

    // KV pool sizing: clamp the len-derived pool to what the GPU can actually
    // hold.  Without this, a large config max_position_embeddings (e.g.
    // Qwen2.5-1.5B = 131072) derives a pool larger than the GPU → OOM.
    //
    // Memory budget modes:
    //  - kv_cache_mb > 0: explicit caller budget (per-layer), capped by the
    //    actual free memory when budget_from_free_ (co-resident draft).
    //  - budget_from_free_ (co-resident draft engine): use the ACTUAL free
    //    memory at construction — the target's weights + KV are already
    //    resident, so "use what's left for the draft's KV" is exactly right.
    //  - default (standalone engine): device TOTAL minus THIS engine's weights
    //    (computed from tensor sizes) minus a safety margin.  Deliberately
    //    avoids cudaMemGetInfo free, which CUDA's caching allocator
    //    under-reports when engines are created/destroyed sequentially (test
    //    suites), which would wrongly shrink the pool to 1 block.
    {
        size_t total_bytes = 0, free_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        const size_t kSafetyMargin = 256ULL * 1024 * 1024;  // 256 MB
        size_t block_bytes = static_cast<size_t>(BLOCK_SIZE) * Hkv_ * hd_ * 2
                             * sizeof(float);  // KV storage is F32

        size_t budget_bytes = 0;
        if (kv_cache_mb_ > 0) {
            budget_bytes = static_cast<size_t>(kv_cache_mb_) * 1048576;
        } else {
            size_t weights_bytes = 0;
            auto add_w = [&](const Tensor& t) {
                weights_bytes += static_cast<size_t>(t.numel()) * dtype_size(t.dtype());
            };
            add_w(embed_w_);
            add_w(final_norm_);
            for (auto& L : layers_) {
                add_w(L->q); add_w(L->k); add_w(L->v); add_w(L->o);
                add_w(L->a_n); add_w(L->gate); add_w(L->up); add_w(L->down);
                add_w(L->m_n);
            }
            if (budget_from_free_) {
                budget_bytes = (free_bytes > kSafetyMargin)
                                   ? (free_bytes - kSafetyMargin) : 0;
            } else {
                budget_bytes = (total_bytes > weights_bytes + kSafetyMargin)
                                   ? (total_bytes - weights_bytes - kSafetyMargin)
                                   : 0;
            }
        }
        // A co-resident engine must never exceed the actual free memory, even
        // if the caller set an explicit kv_cache_mb.
        if (budget_from_free_ && budget_bytes > free_bytes)
            budget_bytes = free_bytes;

        size_t per_layer_pool = budget_bytes / static_cast<size_t>(n_layers_);
        int64_t blocks_by_mem = static_cast<int64_t>(per_layer_pool / block_bytes);
        int64_t blocks_by_len =
            (static_cast<int64_t>(max_seq_len_) + BLOCK_SIZE - 1) / BLOCK_SIZE + 8;
        int64_t chosen = (std::min)(blocks_by_len, blocks_by_mem);
        if (chosen < 1) chosen = 1;
        if (chosen != num_blocks_) {
            num_blocks_ = static_cast<int>(chosen);
            int64_t pool_seq = static_cast<int64_t>(num_blocks_) * BLOCK_SIZE;
            if (static_cast<int64_t>(max_seq_len_) > pool_seq)
                max_seq_len_ = static_cast<int>(pool_seq);
            max_blocks_per_seq_ = (max_seq_len_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
            printf("  KV pool clamped to %d blocks (mem budget: free %.1f MB "
                   "of %.1f MB) -> max_seq_len=%d\n",
                   num_blocks_, free_bytes / 1048576.0,
                   total_bytes / 1048576.0, max_seq_len_);
        }
    }

    // Allocate KV cache pools
    printf("Allocating KV cache pools (%d blocks x %d layers)...\n",
           num_blocks_, n_layers_);
    kv_allocators_.reserve(n_layers_);
    for (int l = 0; l < n_layers_; l++) {
        kv_allocators_.push_back(std::make_unique<kv_cache::BlockAllocator>(
            num_blocks_, BLOCK_SIZE, Hkv_, hd_, prefix_cache_policy_));
    }
    // Use 64-bit arithmetic for the byte counts — the raw int product overflows
    // for large pools (e.g. 1.5B with max_seq_len=131072 → ~7.5 GB > INT_MAX).
    {
        size_t per_layer_pool = static_cast<size_t>(num_blocks_) * BLOCK_SIZE
                                * Hkv_ * hd_ * 2 * sizeof(float);
        printf("KV cache pools ready (%.1f MB per layer, %.1f MB total)\n",
               per_layer_pool / 1048576.0,
               per_layer_pool * n_layers_ / 1048576.0);
    }

    // Pre-allocate batch scratch tensors for batched paged_attention
    // d_all_block_tables_ [max_batch_tokens * max_blocks_per_seq_]
    // (max_batch_tokens is an upper bound on decode sequences per step)
    d_all_block_tables_ = Tensor(
        {max_batch_tokens_ * max_blocks_per_seq_}, DType::I32, Device::CUDA);
    d_all_seq_lens_ = Tensor({max_batch_tokens_}, DType::I32, Device::CUDA);

    // ---- Initialize xgrammar (structured output) ----
    {
        snprintf(buf, sizeof(buf), "%s/tokenizer.json", model_dir.c_str());
        tokenizer::Tokenizer tok(buf);
        std::vector<std::string> vocab(vocab_);
        for (int i = 0; i < vocab_; ++i) {
            vocab[i] = tok.id_to_str(i);
        }
        tokenizer_info_ = std::make_unique<xgrammar::TokenizerInfo>(vocab);
        grammar_compiler_ = std::make_unique<xgrammar::GrammarCompiler>(
            *tokenizer_info_,
            2,     // max_threads
            true   // cache_enabled
        );
        printf("xgrammar: GrammarCompiler ready, vocab_size=%d\n", vocab_);
    }
}

// EngineServer destructor needs xgrammar complete types.
// Defined here (not inline in header) so that TUs that only include engine.h
// don't need the full xgrammar headers.
EngineServer::~EngineServer() = default;

void EngineServer::clear_kv_cache() {
    for (auto& alloc : kv_allocators_) {
        alloc->reset();
    }
    // Also clear draft engine's KV cache if present
    if (draft_engine_) {
        draft_engine_->clear_kv_cache();
    }
}

// ============================================================================
// enable_speculative_decode
// ============================================================================

void EngineServer::enable_speculative_decode(const SpeculativeDecodeConfig& cfg) {
    spec_cfg_ = cfg;
    if (!cfg.draft_model_dir.empty()) {
        // Dual-model mode: a separate draft model owns its own KV cache.  It is
        // created co-resident with the target, so its KV pool is budgeted from
        // the ACTUAL free memory (the target's weights + KV are already
        // resident) — this is what makes "use the remaining memory for the
        // draft's KV" work on small GPUs.
        auto draft = std::make_unique<EngineServer>(
            cfg.draft_model_dir, max_seq_len_, max_batch_tokens_,
            kv_cache_mb_, prefix_cache_policy_, use_fp16_,
            /*budget_from_free=*/true);
        draft_engine_ = std::make_unique<DualModelDraftEngine>(std::move(draft));
        printf("Speculative decoding: draft model loaded from %s\n",
               cfg.draft_model_dir.c_str());
    } else {
        // Self-speculation: the target's first M layers as draft, sharing the
        // target's weights but owning its OWN KV cache (never corrupts the
        // target's authoritative cache).
        int M = cfg.draft_layers > 0 ? cfg.draft_layers : n_layers_ / 4;
        if (M < 1) M = 1;
        if (M > n_layers_) M = n_layers_;
        draft_engine_ = std::make_unique<SelfSpecDraftEngine>(*this, M);
        printf("Speculative decoding: self-speculation mode\n");
    }
    printf("  num_draft_tokens = %d\n", cfg.num_draft_tokens);
    printf("  verify_batch = %d\n", cfg.verify_batch ? 1 : 0);
}

// ============================================================================
// create_request_state
// ============================================================================

// RequestState destructor needs the complete GrammarMatcher type.
// Defined here because engine_server.cpp includes <xgrammar/xgrammar.h>.
RequestState::~RequestState() = default;

std::unique_ptr<RequestState> EngineServer::create_request_state(
    const Request& req, int D)
{
    auto state = std::make_unique<RequestState>();
    state->id = req.id;
    state->prompt_tokens = req.prompt_tokens;
    state->max_new_tokens = req.max_new_tokens;
    state->eos_token_id = req.eos_token_id;
    state->temperature = req.temperature;

    // ---- Initialize GrammarMatcher if request has structured output ----
    if (!req.json_schema.empty()) {
        try {
            auto compiled = grammar_compiler_->CompileJSONSchema(
                req.json_schema);
            state->grammar_matcher = std::make_unique<xgrammar::GrammarMatcher>(
                compiled);
            state->grammar_mask.assign((vocab_ + 31) / 32, 0);
        } catch (const std::exception& e) {
            fprintf(stderr, "Grammar compile failed for req %d: %s\n",
                    req.id, e.what());
        }
    } else if (!req.regex.empty()) {
        try {
            auto compiled = grammar_compiler_->CompileRegex(
                req.regex);
            state->grammar_matcher = std::make_unique<xgrammar::GrammarMatcher>(
                compiled);
            state->grammar_mask.assign((vocab_ + 31) / 32, 0);
        } catch (const std::exception& e) {
            fprintf(stderr, "Regex compile failed for req %d: %s\n",
                    req.id, e.what());
        }
    }

    state->seq_len = 0;
    state->num_prefilled = 0;
    state->num_cached_tokens = 0;
    state->finished = false;

    // Initialise empty block tables for each layer
    state->block_tables.reserve(n_layers_);
    for (int l = 0; l < n_layers_; l++) {
        state->block_tables.emplace_back(max_blocks_per_seq_);
    }

    // Initialise draft engine block tables for speculative decoding
    // (both self-spec layer-slice and dual-model).
    if (draft_engine_) {
        int dl = draft_engine_->num_layers();
        int dmax_blocks = max_seq_len_ / draft_engine_->block_size();
        state->draft_block_tables.reserve(dl);
        for (int l = 0; l < dl; l++) {
            state->draft_block_tables.emplace_back(dmax_blocks);
        }
    }

    // Allocate next_hidden_state (uninitialised until first prefill)
    // Uses weight_dtype_ so that the hidden state dtype matches the weights
    // (fp16 when use_fp16=true, fp32 otherwise).
    state->next_hidden_state = Tensor({1, D}, weight_dtype_, Device::CUDA);

    return state;
}

// ============================================================================
// release_request — free ALL blocks across ALL layers
// ============================================================================

void EngineServer::release_request(RequestState& state) {
    // Release target engine blocks
    if (!state.block_tables.empty()) {
        auto& bt0 = state.block_tables[0];
        for (int i = 0; i < bt0.size(); i++) {
            int block_id = bt0[i];
            for (auto& alloc : kv_allocators_) {
                alloc->release(block_id);
            }
        }
    }
    // Release draft engine blocks (both self-spec and dual-model modes).
    // Draft allocators are owned by the active DraftEngine.
    if (draft_engine_ && !state.draft_block_tables.empty()) {
        auto& dbt0 = state.draft_block_tables[0];
        auto& allocs = draft_engine_->allocators();
        for (int i = 0; i < static_cast<int>(dbt0.size()); i++) {
            int block_id = dbt0[i];
            for (int l = 0; l < static_cast<int>(allocs.size()); l++) {
                allocs[l]->release(block_id);
            }
        }
    }
}

// Simple accessors
int EngineServer::used_blocks() const {
    return num_blocks_ - kv_allocators_[0]->num_free();
}
int EngineServer::free_blocks() const {
    return kv_allocators_[0]->num_free();
}

// ============================================================================
// Block helpers
// ============================================================================

void EngineServer::scatter_prefill_kv(
    int layer, const Tensor& k_contig, const Tensor& v_contig,
    int n_tokens, int start_pos,
    const std::vector<kv_cache::BlockTable>& block_tables)
{
    const float* k_src = k_contig.data<float>();
    const float* v_src = v_contig.data<float>();
    size_t row_bytes = static_cast<size_t>(Hkv_) * hd_ * sizeof(float);
    auto& alloc = *kv_allocators_[layer];
    auto& bt = block_tables[layer];

    for (int i = 0; i < n_tokens; i++) {
        int pos     = start_pos + i;
        int blk_idx = pos / BLOCK_SIZE;
        int offset  = pos % BLOCK_SIZE;
        int phys    = bt[blk_idx];

        void* dst_k = static_cast<char*>(alloc.k_data(phys))
                      + offset * row_bytes;
        void* dst_v = static_cast<char*>(alloc.v_data(phys))
                      + offset * row_bytes;
        const void* src_k = k_src + i * Hkv_ * hd_;
        const void* src_v = v_src + i * Hkv_ * hd_;

        cudaMemcpy(dst_k, src_k, row_bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst_v, src_v, row_bytes, cudaMemcpyDeviceToDevice);
    }
}

void EngineServer::write_decode_kv(
    int layer, int token_pos, const Tensor& k_new, const Tensor& v_new,
    const std::vector<kv_cache::BlockTable>& block_tables,
    kv_cache::BlockAllocator& alloc)
{
    int blk_idx = token_pos / BLOCK_SIZE;
    int offset  = token_pos % BLOCK_SIZE;
    // Bounds check: if the block table lacks this block (KV pool exhausted, or
    // a caller failed to ensure blocks), writing would read a garbage physical
    // id and cudaMemcpy to a wild pointer (SIGSEGV).  Skip gracefully instead.
    if (blk_idx < 0 || blk_idx >= static_cast<int>(block_tables[layer].size())) {
        fprintf(stderr,
                "FATAL: write_decode_kv block OOB layer=%d pos=%d blk=%d/%zu "
                "(KV pool exhausted — skipping write)\n",
                layer, token_pos, blk_idx, block_tables[layer].size());
        return;
    }
    int phys    = block_tables[layer][blk_idx];
    size_t row_bytes = static_cast<size_t>(Hkv_) * hd_ * sizeof(float);

    void* dst_k = static_cast<char*>(alloc.k_data(phys))
                  + offset * row_bytes;
    void* dst_v = static_cast<char*>(alloc.v_data(phys))
                  + offset * row_bytes;

    cudaMemcpy(dst_k, k_new.raw(), row_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst_v, v_new.raw(), row_bytes, cudaMemcpyDeviceToDevice);
}

// ============================================================================
// build_input_tensor — embed prefill tokens + copy decode hidden states
// ============================================================================

Tensor EngineServer::build_input_tensor(
    const ScheduleStep& step,
    const std::unordered_map<int, std::unique_ptr<RequestState>>& states,
    std::vector<std::pair<int, int>>& entry_map)
{
    int total_tokens = step.total_tokens();
    entry_map.clear();
    entry_map.reserve(step.entries.size());

    // Collect all token IDs into one flat vector, then do a single GPU
    // embedding lookup.  This replaces the old GPU→CPU→GPU round-trip
    // (copy entire embed_w_ to CPU, memcpy per token, copy back to GPU).
    std::vector<int> all_token_ids;
    all_token_ids.reserve(total_tokens);

    int row = 0;
    for (const auto& entry : step.entries) {
        int start = row;
        auto it = states.find(entry.request_idx);
        if (it == states.end()) {
            fprintf(stderr, "FATAL: RequestState not found for id=%d\n",
                    entry.request_idx);
            for (int t = 0; t < entry.num_tokens; t++)
                all_token_ids.push_back(0);
            row += entry.num_tokens;
            entry_map.push_back({start, row});
            continue;
        }
        auto& state = *it->second;

        if (entry.is_prefill) {
            int chunk_size = entry.num_tokens;
            int chunk_start = entry.start_pos;
            for (int t = 0; t < chunk_size; t++) {
                int tid = state.prompt_tokens[chunk_start + t];
                all_token_ids.push_back((tid >= 0 && tid < vocab_) ? tid : 0);
            }
            row += chunk_size;
        } else {
            // Decode: embed the token at the actual RoPE position in the
            // sequence.  This matches forward_partial's convention:
            // embed e(t_pos) at RoPE position pos, producing h_pos which
            // predicts t_{pos+1}.
            //
            // When the first decode token was pre-sampled during prefill
            // post-processing, it's already in generated_tokens.  When
            // grammar is active (or pre-sample was skipped for any other
            // reason), fall back to embedding the last known token.
            int pos = state.seq_len;
            int P = static_cast<int>(state.prompt_tokens.size());
            int token_id;
            if (pos < P) {
                token_id = state.prompt_tokens[pos];
            } else {
                int gen_idx = pos - P;
                if (gen_idx < static_cast<int>(state.generated_tokens.size())) {
                    token_id = state.generated_tokens[gen_idx];
                } else if (!state.generated_tokens.empty()) {
                    // Position beyond generated range — use last known token
                    token_id = state.generated_tokens.back();
                } else {
                    // No tokens generated yet (e.g. grammar mode skipped
                    // pre-sample).  Use the last prompt token and use the
                    // shifted convention (old behaviour).
                    token_id = state.prompt_tokens.back();
                }
            }
            all_token_ids.push_back((token_id >= 0 && token_id < vocab_) ? token_id : 0);
            row += 1;
        }

        entry_map.push_back({start, row});
    }

    // Single GPU embedding lookup — no CPU round-trip for the weight matrix.
    Tensor token_ids_gpu({total_tokens}, DType::I32, Device::CUDA);
    token_ids_gpu.copy_from(all_token_ids.data(), total_tokens * sizeof(int));
    return ops::embedding(token_ids_gpu, embed_w_);
}

// ============================================================================
// forward_partial — run a range of layers with paged-attention + KV cache write
// ============================================================================
//
// Runs layers [start_layer, start_layer + num_layers) on a single-token hidden
// state.  Unlike forward_n_layers (pure compute), this function:
//   - Uses paged_attention to attend to *all* prior K/V in the cache
//     PLUS its own K/V (self-attention), matching forward_n_layers' causal attn.
//   - Writes K/V to the cache at `token_pos` BEFORE paged_attention so that
//     the attention (with attn_seq_len = token_pos + 1) includes self-attention.
//   - Applies RoPE at the absolute position `token_pos`
//
// This is the correct decode semantics for speculative draft/verify steps
// where the new token must attend to the full sequence history.

Tensor EngineServer::run_layers(
    const Tensor& hidden,      // [T, D_] — input (weight_dtype_)
    int start_layer,           // first absolute layer index (0-based)
    int num_layers,            // number of layers to run
    int start_pos,             // absolute RoPE position of the first token
    const std::vector<kv_cache::BlockTable>& block_tables_abs,  // per absolute layer
    std::vector<kv_cache::BlockAllocator*>& allocs_abs,          // per absolute layer
    bool write_kv)
{
    int T = hidden.size(0);
    int half_hd = hd_ / 2;
    int max_blocks = max_blocks_per_seq_;

    Tensor h({T, D_}, weight_dtype_, Device::CUDA);
    size_t h_row_bytes = static_cast<size_t>(D_) * dtype_size(weight_dtype_);
    cudaMemcpy(h.raw(), hidden.raw(), static_cast<size_t>(T) * h_row_bytes,
               cudaMemcpyDeviceToDevice);

    // Row i attends to positions 0..start_pos+i-1 (decode convention).
    std::vector<int> seq_lens(T);
    for (int i = 0; i < T; i++) seq_lens[i] = start_pos + i;

    int end_layer = start_layer + num_layers;
    for (int l = start_layer; l < end_layer && l < n_layers_; l++) {
        auto& L = *layers_[l];
        auto& alloc = *allocs_abs[l];

        // --- RMSNorm ---
        Tensor normed = rms_norm(h, L.a_n, cfg_.rms_norm_eps);

        // --- Q / K / V projection (batched) ---
        Tensor q, k, v;
        if (weight_dtype_ == DType::F16) {
            q = gemm_f16f32(normed, L.q, true);
            k = gemm_f16f32(normed, L.k, true);
            v = gemm_f16f32(normed, L.v, true);
        } else {
            q = gemm(normed, L.q, true);
            k = gemm(normed, L.k, true);
            v = gemm(normed, L.v, true);
        }
        q.reshape_inplace({T, Hq_, hd_});
        k.reshape_inplace({T, Hkv_, hd_});
        v.reshape_inplace({T, Hkv_, hd_});

        // --- RoPE at positions [start_pos, start_pos+T-1] ---
        {
            std::vector<float> vc(static_cast<size_t>(T) * half_hd);
            std::vector<float> vs(static_cast<size_t>(T) * half_hd);
            for (int t = 0; t < T; t++) {
                int pos = start_pos + t;
                for (int d = 0; d < half_hd; d++) {
                    float theta = static_cast<float>(pos)
                        / powf(cfg_.rope_theta, 2.0f * d / hd_);
                    size_t idx = static_cast<size_t>(t) * half_hd + d;
                    vc[idx] = cosf(theta);
                    vs[idx] = sinf(theta);
                }
            }
            Tensor cos({T, half_hd}, DType::F32, Device::CUDA);
            Tensor sin({T, half_hd}, DType::F32, Device::CUDA);
            cos.copy_from(vc.data(), cos.nbytes());
            sin.copy_from(vs.data(), sin.nbytes());

            rope(q, &k, cos, sin);
        }
        cudaDeviceSynchronize();

        // --- Write K/V BEFORE attention so later rows see earlier rows' KV ---
        if (write_kv) {
            size_t k_row_bytes = static_cast<size_t>(Hkv_) * hd_ * sizeof(float);
            for (int i = 0; i < T; i++) {
                Tensor k_new({1, Hkv_, hd_}, DType::F32, Device::CUDA);
                cudaMemcpy(k_new.raw(),
                           static_cast<const char*>(k.raw()) + i * k_row_bytes,
                           k_row_bytes, cudaMemcpyDeviceToDevice);
                Tensor v_new({1, Hkv_, hd_}, DType::F32, Device::CUDA);
                cudaMemcpy(v_new.raw(),
                           static_cast<const char*>(v.raw()) + i * k_row_bytes,
                           k_row_bytes, cudaMemcpyDeviceToDevice);
                write_decode_kv(l, start_pos + i, k_new, v_new,
                                block_tables_abs, alloc);
            }
        }

        // --- Batched paged attention (T rows of one sequence) ---
        // Each row repeats the same per-layer block table (same sequence);
        // seq_lens[i] = start_pos+i bounds each row's attention.
        {
            std::vector<int> h_bt(static_cast<size_t>(T) * max_blocks, -1);
            const auto& bt = block_tables_abs[l];
            int bt_len = static_cast<int>(bt.size());
            for (int i = 0; i < T; i++)
                for (int b = 0; b < bt_len && b < max_blocks; b++)
                    h_bt[static_cast<size_t>(i) * max_blocks + b] = bt[b];
            Tensor d_bt({T * max_blocks}, DType::I32, Device::CUDA);
            d_bt.copy_from(h_bt.data(), h_bt.size() * sizeof(int));
            Tensor d_sl({T}, DType::I32, Device::CUDA);
            d_sl.copy_from(seq_lens.data(), T * sizeof(int));

            Tensor attn_out_3d = kv_cache::paged_attention(
                q, d_bt.data<int>(), max_blocks, d_sl.data<int>(), alloc);

            // --- O-projection + residual ---
            Tensor attn_2d = attn_out_3d.view({T, D_});
            if (attn_2d.dtype() != weight_dtype_) {
                if (weight_dtype_ == DType::F16)
                    attn_2d = cast_f32_to_f16(attn_2d);
                else
                    attn_2d = cast_f16_to_f32(attn_2d);
            }
            attn_2d = gemm(attn_2d, L.o, true);
            h = ops::add(h, attn_2d);
        }

        // --- MLP ---
        normed = rms_norm(h, L.m_n, cfg_.rms_norm_eps);
        Tensor gate = gemm(normed, L.gate, true);
        Tensor up   = gemm(normed, L.up,   true);
        silu_inplace(gate);
        Tensor mlp = ops::mul(gate, up);
        mlp = gemm(mlp, L.down, true);
        h = ops::add(h, mlp);
    }

    return h;
}

// ============================================================================
// forward_partial — single-token layer runner (wrapper over run_layers)
// ============================================================================
// All current callers pass attn_seq_len == token_pos, which run_layers encodes
// as the single row's seq_len (row 0 attends to 0..token_pos-1).  The separate
// attn_seq_len parameter is kept for API compatibility but is effectively
// redundant.

Tensor EngineServer::forward_partial(
    const Tensor& hidden,      // [1, D_] — input (weight_dtype_)
    int start_layer,           // first layer index (0-based)
    int num_layers,            // number of layers to run
    const std::vector<kv_cache::BlockTable>& block_tables,
    int token_pos,             // absolute RoPE position + KV write position
    int attn_seq_len)          // (deprecated) == token_pos for all callers
{
    (void)attn_seq_len;
    std::vector<kv_cache::BlockAllocator*> allocs(n_layers_);
    for (int l = 0; l < n_layers_; l++) allocs[l] = kv_allocators_[l].get();
    return run_layers(hidden, start_layer, num_layers, token_pos,
                      block_tables, allocs, /*write_kv=*/true);
}

// ============================================================================
// forward_batched — verify k tokens of one sequence through ALL layers
// ============================================================================
// Uses the same batched kernels as step()'s decode path, so the K/V written is
// numerically consistent with normal decoding (fixes speculative output
// divergence).  Row i occupies position start_pos+i and its KV is written
// before the batched attention, so row j sees rows 0..j-1's KV.

Tensor EngineServer::forward_batched(
    const Tensor& tokens,
    const std::vector<kv_cache::BlockTable>& block_tables,
    int start_pos,
    bool write_kv)
{
    std::vector<kv_cache::BlockAllocator*> allocs(n_layers_);
    for (int l = 0; l < n_layers_; l++) allocs[l] = kv_allocators_[l].get();
    return run_layers(tokens, 0, n_layers_, start_pos,
                      block_tables, allocs, write_kv);
}


// ============================================================================
// speculate() — speculative decoding orchestrator (draft + verify + accept)
// ============================================================================
// Called by BatchMainLoop AFTER engine_.step() so the core decode path stays
// speculation-free.  For each active decode request:
//
//   1. sync  — bring the draft engine's own KV up to state.seq_len
//   2. draft — greedy D1..Dk from the draft engine (never touches target KV)
//   3. verify — one batched forward_batched over [base, D1..D_{k-1}] through
//               the FULL model, writing authoritative target K/V
//   4. accept — keep D_{j+1} while row j's target argmax matches AND grammar
//               permits it; on the first mismatch sample a correction token
//               from the target distribution and overwrite the rejected
//               position's K/V with it
//   5. rollback — drop the draft cache beyond the last accepted position
//
// Position convention: after step(), state.seq_len == L and the base token
// occupies position L (its K/V not yet written).  Row j of the verify input
// sits at position L+j, attends to 0..L+j-1, and predicts D_{j+1} at L+j+1.

void EngineServer::speculate(
    const ScheduleStep& step,
    std::unordered_map<int, std::unique_ptr<RequestState>>& states)
{
    if (!draft_engine_ || !spec_cfg_.enabled) return;
    if (step.empty()) return;
    int k_cfg = spec_cfg_.num_draft_tokens;
    int vocab = vocab_;
    size_t rb = static_cast<size_t>(D_) * dtype_size(weight_dtype_);

    // Embed a batch of token ids -> [T, D].
    auto embed = [&](const std::vector<int>& tokens) {
        int T = static_cast<int>(tokens.size());
        std::vector<int> clamped(T);
        for (int i = 0; i < T; i++)
            clamped[i] = (tokens[i] >= 0 && tokens[i] < vocab) ? tokens[i] : 0;
        Tensor ids({T}, DType::I32, Device::CUDA);
        ids.copy_from(clamped.data(), T * sizeof(int));
        return ops::embedding(ids, embed_w_);
    };

    // Fill the grammar bitmask into state.grammar_mask at the current FSM
    // state.  Returns false if there is no grammar matcher.
    auto fill_mask = [&](RequestState& st) -> bool {
        if (!st.grammar_matcher) return false;
        int64_t mask_shape = static_cast<int64_t>(st.grammar_mask.size());
        DLTensor bitmask_dl;
        bitmask_dl.data        = st.grammar_mask.data();
        bitmask_dl.device      = {kDLCPU, 0};
        bitmask_dl.ndim        = 1;
        bitmask_dl.shape       = &mask_shape;
        bitmask_dl.dtype       = {kDLInt, 32, 1};
        bitmask_dl.strides     = nullptr;
        bitmask_dl.byte_offset = 0;
        st.grammar_matcher->FillNextTokenBitmask(&bitmask_dl);
        return true;
    };
    auto mask_allows = [&](RequestState& st, int token) -> bool {
        return (st.grammar_mask[token / 32] >> (token % 32)) & 1;
    };

    for (const auto& entry : step.entries) {
        if (entry.is_prefill) continue;
        auto it = states.find(entry.request_idx);
        if (it == states.end()) continue;
        auto& state = *it->second;
        if (state.finished) continue;
        if (state.generated_tokens.empty()) continue;  // first_token pre-sampled

        // Grammar completed (e.g. JSON object closed): no further generation.
        if (state.grammar_matcher && state.grammar_matcher->IsTerminated()) {
            state.finished = true;
            continue;
        }

        int L = state.seq_len;
        int k = k_cfg;
        if (L + k >= max_seq_len_) k = max_seq_len_ - L - 1;
        if (k <= 0) {
            // At the sequence-length limit: cannot draft/verify any further.
            state.finished = true;
            continue;
        }

        auto t_step0 = std::chrono::steady_clock::now();

        // ---- 1. Sync + draft ----
        draft_engine_->sync(state);
        auto t_d0 = std::chrono::steady_clock::now();
        std::vector<float> draft_probs;  // full p_d per draft position (when sampling)
        std::vector<int> drafts = draft_engine_->draft(
            state, k, spec_cfg_.probability_acceptance,
            spec_cfg_.probability_acceptance ? &draft_probs : nullptr);
        auto t_d1 = std::chrono::steady_clock::now();
        state.spec_stats.drafted += static_cast<int>(drafts.size());
        state.spec_drafted += static_cast<int>(drafts.size());
        state.spec_stats.draft_ms +=
            std::chrono::duration<double, std::milli>(t_d1 - t_d0).count();
        if (drafts.empty()) continue;

        // ---- 2. Verify input: [base, D1..D_{n-1}] at positions L..L+n-1 ----
        int n = static_cast<int>(drafts.size());
        std::vector<int> verify_input;
        verify_input.reserve(n);
        verify_input.push_back(state.generated_tokens.back());
        for (int i = 0; i < n - 1; i++) verify_input.push_back(drafts[i]);

        // Ensure the target block table covers positions L..L+n-1.
        {
            int blocks_needed = (L + n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            while (static_cast<int>(state.block_tables[0].size()) < blocks_needed) {
                int bid = kv_allocators_[0]->allocate(0);
                if (bid < 0) break;
                for (int l = 1; l < n_layers_; l++) kv_allocators_[l]->allocate(0);
                for (int l = 0; l < n_layers_; l++) state.block_tables[l].append(bid);
            }
        }

        auto t_v0 = std::chrono::steady_clock::now();
        Tensor h_out;
        if (spec_cfg_.verify_batch) {
            // Batched verify (M=k GEMMs): most efficient, but numerically
            // diverges from the sequential single-token (M=1) draft, lowering
            // acceptance even for a same-model draft.
            Tensor h_verify = embed(verify_input);
            h_out = forward_batched(h_verify, state.block_tables, L,
                                    /*write_kv=*/true);
        } else {
            // Sequential verify (M=1 GEMMs per row): numerically matches the
            // draft's single-token path, so acceptance reflects only the
            // draft-model approximation.  Slower (k weight-loads).
            h_out = Tensor({n, D_}, weight_dtype_, Device::CUDA);
            for (int i = 0; i < n; i++) {
                Tensor h_i = embed({verify_input[i]});
                Tensor h_o = forward_batched(h_i, state.block_tables, L + i,
                                             /*write_kv=*/true);
                cudaMemcpy(static_cast<char*>(h_out.raw()) + i * rb,
                           h_o.raw(), rb, cudaMemcpyDeviceToDevice);
            }
        }
        auto t_v1 = std::chrono::steady_clock::now();
        state.spec_stats.verify_ms +=
            std::chrono::duration<double, std::milli>(t_v1 - t_v0).count();

        // ---- 3. Acceptance loop ----
        std::mt19937 rng(42);
        int accepted_count = 0;
        bool finished = false;
        for (int j = 0; j < n && !finished; j++) {
            // Row j at position L+j predicts drafts[j] at L+j+1.
            Tensor h_pos({1, D_}, weight_dtype_, Device::CUDA);
            cudaMemcpy(h_pos.raw(),
                       static_cast<const char*>(h_out.raw()) + j * rb,
                       rb, cudaMemcpyDeviceToDevice);
            Tensor hn = rms_norm(h_pos, final_norm_, cfg_.rms_norm_eps);
            Tensor lg = ops::lm_head_logits(hn, embed_w_);

            // Grammar: re-fill the mask at the CURRENT FSM state (updated by
            // AcceptToken for each previously-accepted draft).
            bool grammar_ok = true;
            if (state.grammar_matcher) {
                fill_mask(state);
                grammar_ok = mask_allows(state, drafts[j]);
            }

            // ---- Decide whether to accept drafts[j] ----
            // probability_acceptance (Leviathan): the draft sampled drafts[j]
            // from p_d; accept with probability min(1, p_t(drafts[j])/p_d(
            // drafts[j])).  The emitted distribution exactly matches p_t.
            // Otherwise: strict argmax equality (deterministic).
            bool accept_this = false;
            std::vector<float> logits_cpu;  // populated when probability path
            std::vector<float> p_t;         // softmax(target logits), prob path
            const bool is_prob = spec_cfg_.probability_acceptance;
            if (grammar_ok && is_prob) {
                logits_cpu.resize(vocab);
                lg.copy_to(logits_cpu.data(), vocab * sizeof(float));
                p_t = logits_cpu;
                detail::softmax_inplace(p_t.data(), vocab);
                size_t pd_off = static_cast<size_t>(j) * vocab;
                float p_dj = (pd_off + drafts[j] < draft_probs.size())
                                 ? draft_probs[pd_off + drafts[j]] : 0.0f;
                float p_tj = p_t[drafts[j]];
                float p_accept = (p_dj > 0.0f) ? std::min(1.0f, p_tj / p_dj) : 0.0f;
                std::uniform_real_distribution<float> u(0.0f, 1.0f);
                accept_this = (p_accept >= 1.0f) || (u(rng) < p_accept);
            } else {
                accept_this = grammar_ok && (ops::argmax(lg) == drafts[j]);
            }

            if (accept_this) {
                // Accept.
                state.generated_tokens.push_back(drafts[j]);
                state.seq_len++;
                state.spec_accepted++;
                state.spec_stats.accepted++;
                accepted_count++;
                if (state.grammar_matcher)
                    state.grammar_matcher->AcceptToken(drafts[j]);
                if (drafts[j] == state.eos_token_id ||
                    static_cast<int>(state.generated_tokens.size())
                        >= state.max_new_tokens) {
                    finished = true;
                    state.finished = true;
                }
            } else {
                // Reject at j: sample a correction token at position L+j+1.
                // - probability path: residual q(t) = max(0, p_t(t) - p_d(t)),
                //   renormalized (Leviathan) — guarantees output == p_t.
                // - argmax path: full target distribution.
                SamplingParams sp;
                sp.temperature = state.temperature;
                sp.seed = 42 + state.seq_len;
                const int32_t* mask_ptr = nullptr;
                if (state.grammar_matcher) {
                    fill_mask(state);
                    mask_ptr = state.grammar_mask.data();
                }
                int correction;
                if (is_prob && !logits_cpu.empty()) {
                    // Build the residual into logits_cpu (unnormalized masses).
                    size_t pd_off = static_cast<size_t>(j) * vocab;
                    float qsum = 0.0f;
                    for (int i = 0; i < vocab; i++) {
                        float pd_i = (pd_off + i < draft_probs.size())
                                         ? draft_probs[pd_off + i] : 0.0f;
                        float q = p_t[i] - pd_i;
                        logits_cpu[i] = (q > 0.0f) ? q : 0.0f;
                        qsum += logits_cpu[i];
                    }
                    if (qsum <= 0.0f) {  // degenerate: fall back to p_t
                        for (int i = 0; i < vocab; i++) logits_cpu[i] = p_t[i];
                    }
                    if (mask_ptr) {  // grammar mask zeroes disallowed tokens
                        for (int i = 0; i < vocab; i++)
                            if (!(mask_ptr[i / 32] & (1u << (i % 32))))
                                logits_cpu[i] = 0.0f;
                    }
                    float s = 0.0f;
                    for (int i = 0; i < vocab; i++) s += logits_cpu[i];
                    if (s > 0.0f)
                        for (int i = 0; i < vocab; i++) logits_cpu[i] /= s;
                    correction = detail::multinomial_draw(logits_cpu.data(),
                                                          vocab, rng);
                } else {
                    if (logits_cpu.empty()) {
                        logits_cpu.resize(vocab);
                        lg.copy_to(logits_cpu.data(), vocab * sizeof(float));
                    }
                    correction = ops::sample(logits_cpu.data(), vocab, sp,
                                             rng, mask_ptr);
                }
                state.generated_tokens.push_back(correction);
                state.seq_len++;
                state.spec_stats.rejected++;
                if (state.grammar_matcher)
                    state.grammar_matcher->AcceptToken(correction);
                if (correction == state.eos_token_id ||
                    static_cast<int>(state.generated_tokens.size())
                        >= state.max_new_tokens) {
                    finished = true;
                    state.finished = true;
                }

                // Overwrite the rejected position's K/V with the correction
                // token (forward_batched had embedded the wrong draft token).
                // Positions beyond are never read (attention is bounded by
                // seq_len) and get overwritten as the sequence grows.
                Tensor h_corr = embed({correction});
                Tensor h_corr_out = forward_batched(h_corr, state.block_tables,
                                                    L + j + 1,
                                                    /*write_kv=*/true);
                (void)h_corr_out;
                break;
            }
        }

        // ---- 4. Rollback the draft cache past the last accepted position ----
        draft_engine_->rollback(state);
        state.spec_stats.steps++;
        state.spec_stats.total_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_step0).count();
    }
}

// ============================================================================
// step() — the core batched forward pass
// ============================================================================

std::vector<SampledToken> EngineServer::step(
    const ScheduleStep& step,
    std::unordered_map<int, std::unique_ptr<RequestState>>& states)
{
    std::vector<SampledToken> results;

    if (step.empty()) return results;

    // ------------------------------------------------------------------
    // 0.  SPLIT ENTRIES
    // ------------------------------------------------------------------
    std::vector<int> prefill_indices;
    std::vector<int> decode_indices;
    std::vector<std::pair<int, int>> entry_map;

    for (int i = 0; i < static_cast<int>(step.entries.size()); i++) {
        if (step.entries[i].is_prefill)
            prefill_indices.push_back(i);
        else
            decode_indices.push_back(i);
    }

    int total_tokens = step.total_tokens();
    int groups = Hq_ / Hkv_;
    float attn_scale = 1.0f / sqrtf(static_cast<float>(hd_));

    // ------------------------------------------------------------------
    // 1.  BLOCK ALLOCATION (before the layer loop)
    // ------------------------------------------------------------------

    // Prefill block allocation.
    //
    // When a long prompt is split across multiple entries in the SAME batch
    // (e.g., chunk_size=16, prompt=20 tokens gives two entries P=16 then P=4),
    // the second entry must know its total includes the first entry's tokens
    // so it allocates enough blocks.  We track per-request pf_tokens_in_step
    // here because state.num_prefilled is only updated after the layer loop
    // (post-processing), not during block allocation.
    {
        std::unordered_map<int, int> pf_tokens_in_step;
        for (int idx : prefill_indices) {
            const auto& entry = step.entries[idx];
            auto it = states.find(entry.request_idx);
            if (it == states.end()) continue;
            auto& state = *it->second;

            int req_id = entry.request_idx;
            int prev_in_step = pf_tokens_in_step[req_id];  // 0 if new

            if (state.num_prefilled == 0 && prev_in_step == 0) {
                // ---- First prefill entry for this request: prefix cache ----

                // 1. Look up longest prefix match (layer 0 as authority)
                auto result = kv_allocators_[0]->find_cached_prefix(
                    state.prompt_tokens, 0);
                state.num_cached_tokens = result.matched_tokens;

                // 2. Adopt shared blocks (increment ref_count on all layers)
                for (int bid : result.matched_block_ids) {
                    for (auto& alloc : kv_allocators_) {
                        alloc->increment_ref(bid);
                    }
                    for (int l = 0; l < n_layers_; l++) {
                        state.block_tables[l].append(bid);
                    }
                }

                // 3. Mark cached tokens as prefilled (attention split uses this)
                state.num_prefilled = state.num_cached_tokens;

                // 4. Allocate fresh blocks for the unmatched suffix
                int total_after = state.num_cached_tokens + entry.num_tokens;
                int blocks_needed = (total_after + BLOCK_SIZE - 1) / BLOCK_SIZE;
                int current_blocks = state.num_cached_tokens / BLOCK_SIZE;

                for (int b = current_blocks; b < blocks_needed; b++) {
                    int block_id = kv_allocators_[0]->allocate(0);
                    if (block_id < 0) {
                        fprintf(stderr,
                                "FATAL: KV cache pool exhausted during prefill "
                                "(req %d, layer 0)\n", req_id);
                        break;
                    }
                    for (int l = 1; l < n_layers_; l++) {
                        kv_allocators_[l]->allocate(0);
                    }
                    for (int l = 0; l < n_layers_; l++) {
                        state.block_tables[l].append(block_id);
                    }
                }
            } else {
                // ---- Subsequent prefill entry (same-step split OR later step) ----
                int total_after = state.num_prefilled + prev_in_step
                                 + entry.num_tokens;
                int blocks_needed = (total_after + BLOCK_SIZE - 1) / BLOCK_SIZE;
                int current_blocks = state.block_tables.empty() ? 0
                                     : state.block_tables[0].size();

                for (int b = current_blocks; b < blocks_needed; b++) {
                    int block_id = kv_allocators_[0]->allocate(0);
                    if (block_id < 0) {
                        fprintf(stderr,
                                "FATAL: KV cache pool exhausted during prefill "
                                "(req %d, layer 0)\n", req_id);
                        break;
                    }
                    for (int l = 1; l < n_layers_; l++) {
                        kv_allocators_[l]->allocate(0);
                    }
                    for (int l = 0; l < n_layers_; l++) {
                        state.block_tables[l].append(block_id);
                    }
                }
            }

            pf_tokens_in_step[req_id] += entry.num_tokens;
        }
    }

    // Decode block allocation (on boundary crossing)
    for (int idx : decode_indices) {
        const auto& entry = step.entries[idx];
        auto it = states.find(entry.request_idx);
        if (it == states.end()) continue;
        auto& state = *it->second;

        if (state.seq_len % BLOCK_SIZE == 0) {
            int block_id = kv_allocators_[0]->allocate(0);
            if (block_id < 0) {
                fprintf(stderr,
                        "FATAL: KV cache pool exhausted during decode "
                        "(req %d)\n", entry.request_idx);
                continue;
            }
            for (int l = 1; l < n_layers_; l++) {
                kv_allocators_[l]->allocate(0);
            }
            for (int l = 0; l < n_layers_; l++) {
                state.block_tables[l].append(block_id);
            }
        }
    }

    // Pre-allocate blocks for speculative decode draft tokens.
    // speculative_step() may add up to K tokens to the sequence;
    // guarantee blocks exist for all K positions before the verify
    // phase tries to write K/V data there.
    if (spec_cfg_.enabled) {
        for (int idx : decode_indices) {
            const auto& entry = step.entries[idx];
            auto it = states.find(entry.request_idx);
            if (it == states.end()) continue;
            auto& st = *it->second;
            int future_len = st.seq_len + spec_cfg_.num_draft_tokens;
            int blocks_needed = (future_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
            while (static_cast<int>(st.block_tables[0].size()) < blocks_needed) {
                int bid = kv_allocators_[0]->allocate(0);
                if (bid < 0) break;
                for (int l = 1; l < n_layers_; l++) {
                    kv_allocators_[l]->allocate(0);
                }
                for (int l = 0; l < n_layers_; l++) {
                    st.block_tables[l].append(bid);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 2.  BUILD FLAT INPUT TENSOR  [total_tokens, D]
    // ------------------------------------------------------------------
    Tensor h = build_input_tensor(step, states, entry_map);

    // ------------------------------------------------------------------
    // 3.  TRANSFORMER LAYERS  (0 .. n_layers_-1)
    // ------------------------------------------------------------------
    for (int l = 0; l < n_layers_; l++) {
        auto& L = *layers_[l];

        // --- 3a. RMSNorm (batched over all tokens) ---
        Tensor normed = rms_norm(h, L.a_n, cfg_.rms_norm_eps);

        // --- 3b. Q/K/V projection (batched GEMM) ---
        // In fp16 mode: use gemm_f16f32 to produce F32 outputs for the
        // attention kernels (which require float*).  In fp32 mode: use
        // the standard gemm() which auto-detects dtype from inputs.
        Tensor q_flat, k_flat, v_flat;
        if (weight_dtype_ == DType::F16) {
            q_flat = gemm_f16f32(normed, L.q, true);  // F16+F16 -> F32
            k_flat = gemm_f16f32(normed, L.k, true);
            v_flat = gemm_f16f32(normed, L.v, true);
        } else {
            q_flat = gemm(normed, L.q, true);  // [T, Hq*hd]
            k_flat = gemm(normed, L.k, true);  // [T, Hkv*hd]
            v_flat = gemm(normed, L.v, true);  // [T, Hkv*hd]
        }

        q_flat.reshape_inplace({total_tokens, Hq_, hd_});
        k_flat.reshape_inplace({total_tokens, Hkv_, hd_});
        v_flat.reshape_inplace({total_tokens, Hkv_, hd_});

        // --- 3c. RoPE (per-position, batched over all tokens) ---
        // Build cos/sin with correct absolute positions for each token
        {
            int half_hd = hd_ / 2;
            std::vector<float> vc(total_tokens * half_hd);
            std::vector<float> vs(total_tokens * half_hd);
            int row_off = 0;
            for (const auto& entry : step.entries) {
                // Determine RoPE position for each token in this entry.
                // For prefill: start_pos is correct (0, chunk_size, 2*chunk_size, ...).
                // For decode: scheduler's start_pos is stale (always P).  Use the
                // actual sequence length from RequestState instead.
                int entry_start_pos = entry.start_pos;
                if (!entry.is_prefill) {
                    auto it = states.find(entry.request_idx);
                    if (it != states.end()) {
                        entry_start_pos = it->second->seq_len;
                    }
                }
                for (int t = 0; t < entry.num_tokens; t++) {
                    int pos = entry_start_pos + t;
                    for (int d = 0; d < half_hd; d++) {
                        float theta = static_cast<float>(pos)
                            / powf(cfg_.rope_theta, 2.f * d / hd_);
                        int idx = (row_off + t) * half_hd + d;
                        vc[idx] = cosf(theta);
                        vs[idx] = sinf(theta);
                    }
                }
                row_off += entry.num_tokens;
            }
            Tensor cos({total_tokens, half_hd}, DType::F32, Device::CUDA);
            Tensor sin({total_tokens, half_hd}, DType::F32, Device::CUDA);
            cos.copy_from(vc.data(), cos.nbytes());
            sin.copy_from(vs.data(), sin.nbytes());

            rope(q_flat, &k_flat, cos, sin);
        }
        cudaDeviceSynchronize();

        // --- 3d. ATTENTION ---
        // Flat output for attention: [total_tokens, Hq, hd]
        Tensor attn_flat({total_tokens, Hq_, hd_}, DType::F32, Device::CUDA);
        float* attn_ptr = attn_flat.data<float>();
        const float* q_ptr = q_flat.data<float>();
        const float* k_ptr = k_flat.data<float>();
        const float* v_ptr = v_flat.data<float>();

        // ----- 3d-i. SPLIT PREFILL into first-prefill and chunked-prefill -----
        // First-prefill (historical==0): K/V contiguous, batch entirely.
        // Chunked-prefill (historical>0): needs paged attention, keep per-entry.

        std::vector<int> first_prefill_idxs;
        std::vector<int> chunked_prefill_idxs;
        for (int idx : prefill_indices) {
            auto it = states.find(step.entries[idx].request_idx);
            if (it == states.end()) continue;
            if (it->second->num_prefilled == 0)
                first_prefill_idxs.push_back(idx);
            else
                chunked_prefill_idxs.push_back(idx);
        }

        // ===== Batched first-prefill (common case) =====
        if (!first_prefill_idxs.empty()) {
            int N = static_cast<int>(first_prefill_idxs.size());

            // Build per-entry metadata
            std::vector<int> offsets(N);
            std::vector<int> kv_offsets(N);
            std::vector<int> num_tokens(N);
            std::vector<int> start_poss(N);
            std::vector<int> token_cumsum(N + 1);
            token_cumsum[0] = 0;
            int P_total = 0;

            std::vector<int> bt_flat(N * max_blocks_per_seq_, -1);

            for (int j = 0; j < N; j++) {
                int idx = first_prefill_idxs[j];
                const auto& entry = step.entries[idx];
                auto [start, end] = entry_map[idx];
                auto& state = *states[entry.request_idx];

                offsets[j]    = start * Hq_ * hd_;
                kv_offsets[j] = start * Hkv_ * hd_;
                num_tokens[j] = entry.num_tokens;
                start_poss[j] = entry.start_pos;
                P_total += entry.num_tokens;
                token_cumsum[j + 1] = token_cumsum[j] + entry.num_tokens;

                auto& bt = state.block_tables[l];
                for (int b = 0; b < bt.size(); b++)
                    bt_flat[j * max_blocks_per_seq_ + b] = bt[b];
            }

            // Ensure device buffers are large enough
            if (N > prefill_batch_capacity_) {
                prefill_batch_capacity_ = std::max(N, prefill_batch_capacity_ * 2);
                int cap = prefill_batch_capacity_;
                d_prefill_offsets_       = Tensor({cap}, DType::I32, Device::CUDA);
                d_prefill_kv_offsets_    = Tensor({cap}, DType::I32, Device::CUDA);
                d_prefill_num_tokens_    = Tensor({cap}, DType::I32, Device::CUDA);
                d_prefill_start_poss_    = Tensor({cap}, DType::I32, Device::CUDA);
                d_prefill_token_cumsum_  = Tensor({cap + 1}, DType::I32, Device::CUDA);
                d_prefill_bt_flat_ = Tensor({cap * max_blocks_per_seq_},
                                             DType::I32, Device::CUDA);
            }

            // Upload metadata to device (H2D copies, 6 arrays, O(N) total)
            cudaMemcpy(d_prefill_offsets_.raw(),
                       offsets.data(), N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_prefill_kv_offsets_.raw(),
                       kv_offsets.data(), N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_prefill_num_tokens_.raw(),
                       num_tokens.data(), N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_prefill_start_poss_.raw(),
                       start_poss.data(), N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_prefill_token_cumsum_.raw(),
                       token_cumsum.data(), (N + 1) * sizeof(int),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_prefill_bt_flat_.raw(),
                       bt_flat.data(), bt_flat.size() * sizeof(int),
                       cudaMemcpyHostToDevice);

            // ONE scatter kernel for all first-prefill K/V -> paged blocks
            // DType::F32 — q_flat/k_flat/v_flat are always F32 from gemm/gemm_f16f32.
            kv_cache::scatter_prefill_kv_batched_gpu_dispatch(
                DType::F32,
                k_ptr, v_ptr,
                Hkv_, hd_,
                d_prefill_kv_offsets_.data<int>(),
                d_prefill_num_tokens_.data<int>(),
                d_prefill_start_poss_.data<int>(),
                d_prefill_token_cumsum_.data<int>(),
                d_prefill_bt_flat_.data<int>(),
                max_blocks_per_seq_,
                N, P_total,
                *kv_allocators_[l]);

            // ONE attention kernel: contiguous K/V, writes to attn_flat
            float scale = 1.0f / sqrtf(static_cast<float>(hd_));
            kv_cache::first_prefill_attn_batched_gpu_dispatch(
                DType::F32,
                attn_ptr, q_ptr, k_ptr, v_ptr,
                Hq_, Hkv_, hd_,
                d_prefill_offsets_.data<int>(),
                d_prefill_kv_offsets_.data<int>(),
                d_prefill_num_tokens_.data<int>(),
                d_prefill_token_cumsum_.data<int>(),
                d_prefill_start_poss_.data<int>(),
                N, P_total, scale);
        }

        // ===== Chunked prefill (historical > 0, rare) — per-entry =====
        for (int idx : chunked_prefill_idxs) {
            const auto& entry = step.entries[idx];
            auto [start, end] = entry_map[idx];
            int chunk_size = entry.num_tokens;
            auto it = states.find(entry.request_idx);
            if (it == states.end()) continue;
            auto& state = *it->second;

            int historical = state.num_prefilled;
            int total_seq = historical + chunk_size;

            // Extract Q slice: [chunk_size, Hq, hd]
            Tensor q_i({chunk_size, Hq_, hd_}, DType::F32, Device::CUDA);
            cudaMemcpy(q_i.raw(),
                       q_ptr + start * Hq_ * hd_,
                       chunk_size * Hq_ * hd_ * sizeof(float),
                       cudaMemcpyDeviceToDevice);

            // Extract K slice: [chunk_size, Hkv, hd]
            Tensor k_i({chunk_size, Hkv_, hd_}, DType::F32, Device::CUDA);
            cudaMemcpy(k_i.raw(),
                       k_ptr + start * Hkv_ * hd_,
                       chunk_size * Hkv_ * hd_ * sizeof(float),
                       cudaMemcpyDeviceToDevice);

            // Extract V slice: [chunk_size, Hkv, hd]
            Tensor v_i({chunk_size, Hkv_, hd_}, DType::F32, Device::CUDA);
            cudaMemcpy(v_i.raw(),
                       v_ptr + start * Hkv_ * hd_,
                       chunk_size * Hkv_ * hd_ * sizeof(float),
                       cudaMemcpyDeviceToDevice);

            // Build device-side block table for this sequence.
            int bt_len = state.block_tables[l].size();
            std::vector<int> h_bt(max_blocks_per_seq_, -1);
            for (int i = 0; i < bt_len; i++)
                h_bt[i] = state.block_tables[l][i];
            Tensor d_bt({max_blocks_per_seq_}, DType::I32, Device::CUDA);
            d_bt.copy_from(h_bt.data(), max_blocks_per_seq_ * sizeof(int));

            kv_cache::scatter_prefill_kv_gpu(
                k_i, v_i,
                entry.start_pos, chunk_size,
                d_bt.data<int>(), max_blocks_per_seq_,
                *kv_allocators_[l]);

            Tensor d_seq_len({1}, DType::I32, Device::CUDA);
            d_seq_len.copy_from(&total_seq, sizeof(int));

            Tensor a_out = kv_cache::prefill_paged_attention(
                q_i,
                d_bt.data<int>(),
                max_blocks_per_seq_,
                d_seq_len.data<int>(),
                *kv_allocators_[l],
                false);

            cudaMemcpy(attn_ptr + start * Hq_ * hd_,
                       a_out.raw(),
                       chunk_size * Hq_ * hd_ * sizeof(float),
                       cudaMemcpyDeviceToDevice);
        }

        // ----- 3d-ii. Decode entries (batched paged_attention) -----
        if (!decode_indices.empty()) {
            int N = static_cast<int>(decode_indices.size());

            // Build Q_all: [N, Hq, hd]
            Tensor q_all({N, Hq_, hd_}, DType::F32, Device::CUDA);
            std::vector<RequestState*> decode_states;
            decode_states.reserve(N);

            for (int j = 0; j < N; j++) {
                int idx = decode_indices[j];
                auto [start, end] = entry_map[idx];
                auto it = states.find(step.entries[idx].request_idx);
                if (it == states.end()) continue;
                decode_states.push_back(it->second.get());

                // Copy Q[j] from q_flat
                cudaMemcpy(q_all.data<float>() + j * Hq_ * hd_,
                           q_ptr + start * Hq_ * hd_,
                           Hq_ * hd_ * sizeof(float),
                           cudaMemcpyDeviceToDevice);
            }

            // Build flat block table: [N * max_blocks_per_seq_]
            std::vector<int> bt_flat(N * max_blocks_per_seq_, -1);
            std::vector<int> lens(N);

            for (int j = 0; j < N; j++) {
                auto& state = *decode_states[j];
                auto& bt = state.block_tables[l];
                for (int b = 0; b < bt.size(); b++) {
                    bt_flat[j * max_blocks_per_seq_ + b] = bt[b];
                }
                lens[j] = state.seq_len;
            }

            // Copy metadata to GPU pre-allocated tensors
            d_all_block_tables_.copy_from(bt_flat.data(),
                                          bt_flat.size() * sizeof(int));
            d_all_seq_lens_.copy_from(lens.data(),
                                      N * sizeof(int));

            // ONE kernel call for all decode entries
            Tensor a_out = kv_cache::paged_attention(
                q_all,
                d_all_block_tables_.data<int>(),
                max_blocks_per_seq_,
                d_all_seq_lens_.data<int>(),
                *kv_allocators_[l]);

            // Copy output back to flat tensor and write decode K/V
            for (int j = 0; j < N; j++) {
                int idx = decode_indices[j];
                auto [start, end] = entry_map[idx];
                auto& state = *decode_states[j];

                // Copy attention output row j to flat tensor
                cudaMemcpy(attn_ptr + start * Hq_ * hd_,
                           a_out.data<float>() + j * Hq_ * hd_,
                           Hq_ * hd_ * sizeof(float),
                           cudaMemcpyDeviceToDevice);

                // Write single-token K/V into paged block
                // (block already allocated in step 1 if seq_len crossed
                //  boundary; safety-check the block index is valid)
                int token_pos = state.seq_len;
                int blk_idx = token_pos / BLOCK_SIZE;
                if (blk_idx < state.block_tables[l].size()) {
                    // Extract K_new [1, Hkv, hd] from k_flat
                    Tensor k_new({1, Hkv_, hd_}, DType::F32, Device::CUDA);
                    cudaMemcpy(k_new.raw(),
                               k_ptr + start * Hkv_ * hd_,
                               Hkv_ * hd_ * sizeof(float),
                               cudaMemcpyDeviceToDevice);

                    // Extract V_new [1, Hkv, hd] from v_flat
                    Tensor v_new({1, Hkv_, hd_}, DType::F32, Device::CUDA);
                    cudaMemcpy(v_new.raw(),
                               v_ptr + start * Hkv_ * hd_,
                               Hkv_ * hd_ * sizeof(float),
                               cudaMemcpyDeviceToDevice);

                    write_decode_kv(l, token_pos, k_new, v_new,
                                    state.block_tables, *kv_allocators_[l]);
                }
            }
        }

        // --- 3e. O-projection + residual (batched) ---
        // Ensure attention output dtype matches weight_dtype_ before O-projection
        // GEMM.  In the normal case attn_flat is F32 (Q/K/V are always F32 from
        // gemm/gemm_f16f32), so when weight_dtype_ is F16 we cast down here.
        // The reverse (F16→F32) is also handled for completeness.
        Tensor attn_out_2d = attn_flat.view({total_tokens, D_});
        if (attn_out_2d.dtype() != weight_dtype_) {
            if (weight_dtype_ == DType::F16) {
                attn_out_2d = cast_f32_to_f16(attn_out_2d);
            } else if (weight_dtype_ == DType::F32) {
                attn_out_2d = cast_f16_to_f32(attn_out_2d);
            }
        }
        attn_out_2d = gemm(attn_out_2d, L.o, true);
        h = ops::add(h, attn_out_2d);

        // --- 3f. MLP (batched) ---
        normed = rms_norm(h, L.m_n, cfg_.rms_norm_eps);
        Tensor gate = gemm(normed, L.gate, true);
        Tensor up   = gemm(normed, L.up, true);
        silu_inplace(gate);
        Tensor mlp = ops::mul(gate, up);
        mlp = gemm(mlp, L.down, true);
        h = ops::add(h, mlp);
    }

    // ------------------------------------------------------------------
    // 3.5  PREFIX CACHE INSERTION (after all layers have K/V data)
    // ------------------------------------------------------------------
    for (int idx : prefill_indices) {
        const auto& entry = step.entries[idx];
        auto it = states.find(entry.request_idx);
        if (it == states.end()) continue;
        auto& state = *it->second;

        // Determine which blocks were fully filled in this step.
        int last_written_pos = entry.start_pos + entry.num_tokens - 1;
        int fully_filled_blocks =
            (last_written_pos + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Blocks [0..num_cached_blocks-1] are already in the prefix cache.
        int num_cached_blocks = state.num_cached_tokens / BLOCK_SIZE;
        if (fully_filled_blocks > num_cached_blocks) {
            std::vector<int> new_block_ids;
            for (int i = num_cached_blocks; i < fully_filled_blocks; i++) {
                new_block_ids.push_back(state.block_tables[0][i]);
            }

            int insert_start = num_cached_blocks * BLOCK_SIZE;
            for (auto& alloc : kv_allocators_) {
                alloc->insert_cached_blocks(
                    state.prompt_tokens, insert_start, new_block_ids);
            }

            // Advance cursor so subsequent steps don't re-insert.
            state.num_cached_tokens = fully_filled_blocks * BLOCK_SIZE;
        }
    }
    // ------------------------------------------------------------------
    // 4.  POST-PROCESSING
    // ------------------------------------------------------------------

    // Save pre-final_norm hidden states for ALL decode entries.
    // These are the raw hidden states (before final_norm) of each decode
    // entry's input token — used to populate next_hidden_state and as
    // context for speculative_step when drafting the next K tokens.
    std::unordered_map<int, Tensor> pre_norm_hiddens;
    for (int j = 0; j < static_cast<int>(decode_indices.size()); j++) {
        int idx = decode_indices[j];
        auto [start, end] = entry_map[idx];
        auto it = states.find(step.entries[idx].request_idx);
        if (it == states.end()) continue;
        size_t rbytes = D_ * dtype_size(weight_dtype_);
        Tensor h_pre({1, D_}, weight_dtype_, Device::CUDA);
        cudaMemcpy(h_pre.raw(),
                   static_cast<const char*>(h.raw()) + start * rbytes,
                   rbytes, cudaMemcpyDeviceToDevice);
        pre_norm_hiddens[it->first] = std::move(h_pre);
    }

    h = rms_norm(h, final_norm_, cfg_.rms_norm_eps);

    // Prefill entries: update state (no output tokens)
    for (int idx : prefill_indices) {
        const auto& entry = step.entries[idx];
        auto [start, end] = entry_map[idx];
        auto it = states.find(entry.request_idx);
        if (it == states.end()) continue;
        auto& state = *it->second;

        // Store the last token's hidden state for the first decode step
        size_t row_bytes = D_ * dtype_size(weight_dtype_);
        Tensor last_h({1, D_}, weight_dtype_, Device::CUDA);
        cudaMemcpy(last_h.raw(),
                   static_cast<const char*>(h.raw()) + (end - 1) * row_bytes,
                   row_bytes,
                   cudaMemcpyDeviceToDevice);

        state.next_hidden_state = std::move(last_h);
        state.num_prefilled += entry.num_tokens;

        // Pre-sample the first decode token for entries that just completed
        // their last prefill chunk.  Push it to generated_tokens immediately
        // so that build_input_tensor() embeds it at the correct RoPE position
        // on the next step (matching forward_partial convention).
        if (state.max_new_tokens > 0
            && state.num_prefilled == static_cast<int>(state.prompt_tokens.size())
            && state.generated_tokens.empty()) {

            Tensor h_n = rms_norm(state.next_hidden_state, final_norm_,
                                  cfg_.rms_norm_eps);
            Tensor lg = ops::lm_head_logits(h_n, embed_w_);
            int first_token;
            if (state.grammar_matcher) {
                // Apply grammar mask, then argmax.
                std::vector<float> lc(vocab_);
                lg.copy_to(lc.data(), vocab_ * sizeof(float));
                int64_t mask_shape = static_cast<int64_t>(state.grammar_mask.size());
                DLTensor bitmask_dl;
                bitmask_dl.data        = state.grammar_mask.data();
                bitmask_dl.device      = {kDLCPU, 0};
                bitmask_dl.ndim        = 1;
                bitmask_dl.shape       = &mask_shape;
                bitmask_dl.dtype       = {kDLInt, 32, 1};
                bitmask_dl.strides     = nullptr;
                bitmask_dl.byte_offset = 0;
                state.grammar_matcher->FillNextTokenBitmask(&bitmask_dl);
                int best = 0;
                int sz = static_cast<int>(state.grammar_mask.size());
                for (int i = 0; i < vocab_; i++) {
                    if (state.grammar_mask[i / 32] & (1u << (i % 32))) {
                        if (lc[i] > lc[best]) best = i;
                    }
                }
                first_token = best;
                state.grammar_matcher->AcceptToken(first_token);
            } else {
                first_token = ops::argmax(lg);
            }
            state.generated_tokens.push_back(first_token);
            if (static_cast<int>(state.generated_tokens.size())
                >= state.max_new_tokens) {
                state.finished = true;
            }
        }

        state.seq_len = state.num_prefilled;
    }

    // Decode entries: sample next token, update state, return results
    std::mt19937 rng(42);

    for (int j = 0; j < static_cast<int>(decode_indices.size()); j++) {
        int idx = decode_indices[j];
        const auto& entry = step.entries[idx];
        auto [start, end] = entry_map[idx];
        auto it = states.find(entry.request_idx);
        if (it == states.end()) continue;
        auto& state = *it->second;

        // If the grammar has already terminated (JSON complete), skip
        // further generation for this request and mark it finished.
        if (state.grammar_matcher
            && state.grammar_matcher->IsTerminated()) {
            state.finished = true;
            results.push_back(SampledToken{
                state.id,
                state.eos_token_id,
                true  // is_eos
            });
            continue;
        }

        // Extract this token's hidden state (row `start`, only 1 row)
        // lm_head_logits has a fp16 fast-path (gemm_f16f32) when both
        // tok_h and embed_w_ are F16 — which is naturally the case when
        // use_fp16_ is true (both use weight_dtype_).
        size_t row_bytes = D_ * dtype_size(weight_dtype_);
        Tensor tok_h({1, D_}, weight_dtype_, Device::CUDA);
        cudaMemcpy(tok_h.raw(),
                   static_cast<const char*>(h.raw()) + start * row_bytes,
                   row_bytes,
                   cudaMemcpyDeviceToDevice);

        // LM head -> logits on GPU
        Tensor logits_gpu = ops::lm_head_logits(tok_h, embed_w_);
        std::vector<float> logits_cpu(vocab_);
        logits_gpu.copy_to(logits_cpu.data(), vocab_ * sizeof(float));

        // Sample — temperature is a per-request knob.  When a grammar is
        // active the mask constrains valid tokens; temperature controls
        // sampling within the mask independently (0.0 = greedy).
        SamplingParams sp;
        sp.temperature = state.temperature;
        sp.seed = 42 + state.seq_len;

        const int32_t* mask_ptr = nullptr;
        if (state.grammar_matcher) {
            // Build DLTensor wrapper around our pre-allocated bitmask buffer.
            int64_t mask_shape = static_cast<int64_t>(state.grammar_mask.size());
            DLTensor bitmask_dl;
            bitmask_dl.data        = state.grammar_mask.data();
            bitmask_dl.device      = {kDLCPU, 0};
            bitmask_dl.ndim        = 1;
            bitmask_dl.shape       = &mask_shape;
            bitmask_dl.dtype       = {kDLInt, 32, 1};
            bitmask_dl.strides     = nullptr;
            bitmask_dl.byte_offset = 0;

            bool constrains = state.grammar_matcher->FillNextTokenBitmask(
                &bitmask_dl);
            if (constrains) {
                mask_ptr = state.grammar_mask.data();
            }
        }

        int next_token = ops::sample(logits_cpu.data(), vocab_, sp, rng, mask_ptr);

        // Advance grammar FSM; if it reaches terminal state after this
        // token (e.g. `}` closed the JSON object), mark finished so the
        // next step skips this request cleanly.
        if (state.grammar_matcher) {
            state.grammar_matcher->AcceptToken(next_token);
            if (state.grammar_matcher->IsTerminated()) {
                state.finished = true;
            }
        }

        // Update state
        state.generated_tokens.push_back(next_token);
        state.seq_len += 1;
        // Save pre-final-norm hidden state (for speculative_step / future use).
        // tok_h holds the post-final-norm hidden state (correct for lm_head),
        // but next_hidden_state should store the pre-norm state so that
        // speculative_step() receives a raw hidden state consistent with what
        // forward_n_layers() expects as input.
        {
            auto pn_it = pre_norm_hiddens.find(state.id);
            if (pn_it != pre_norm_hiddens.end()) {
                size_t rbytes = D_ * dtype_size(weight_dtype_);
                Tensor copy({1, D_}, weight_dtype_, Device::CUDA);
                cudaMemcpy(copy.raw(), pn_it->second.raw(),
                           rbytes, cudaMemcpyDeviceToDevice);
                state.next_hidden_state = std::move(copy);
            }
        }

        // Check termination
        bool is_eos = (next_token == state.eos_token_id) ||
                      (static_cast<int>(state.generated_tokens.size())
                       >= state.max_new_tokens);

        if (is_eos) {
            state.finished = true;
        }

        results.push_back(SampledToken{
            state.id,
            next_token,
            is_eos
        });
    }

    // Speculative decoding is orchestrated separately (EngineServer::speculate)
    // by BatchMainLoop AFTER this call, so the core decode path stays clean.
    return results;
}

// ============================================================================
// forward_n_layers — pure compute, no KV cache interaction
// ============================================================================
//
// Runs the first `n` transformer layers on the input hidden states `h`.
// Each layer visits: RMSNorm → Q/K/V GEMM → RoPE → attention → O-proj →
// residual → RMSNorm → SwiGLU MLP → residual.
//
// This is a pure function: it writes nothing to the KV cache, does not
// interact with any BlockAllocator, and does not modify any RequestState.
// It is used by the speculative decoder for both draft generation (n=2)
// and verification (n=n_layers).
//
// The attention is causal (ops::attention computes causal self-attention),
// so each token can only attend to tokens at positions <= its own.  This
// is the same attention function used by the transformer model.

Tensor EngineServer::forward_n_layers(int n,
                                       const Tensor& h,
                                       int start_pos) const {
    // Clamp n to valid range
    if (n < 1) n = 1;
    if (n > n_layers_) n = n_layers_;

    int T = h.size(0);
    int half_hd = hd_ / 2;

    // Copy input to a writable tensor (Tensor is move-only)
    Tensor x({T, D_}, weight_dtype_, Device::CUDA);
    cudaMemcpy(x.raw(), h.raw(), static_cast<size_t>(T) * D_ * dtype_size(weight_dtype_),
               cudaMemcpyDeviceToDevice);

    for (int l = 0; l < n; l++) {
        auto& L = *layers_[l];

        // ---------------------------------------------------------------
        // 1. Pre-attention RMSNorm
        // ---------------------------------------------------------------
        Tensor normed = rms_norm(x, L.a_n, cfg_.rms_norm_eps);

        // ---------------------------------------------------------------
        // 2. Q / K / V projections
        // ---------------------------------------------------------------
        // In FP16 mode we use gemm_f16f32 which produces F32 output
        // tensors (the attention kernels require float*).  In FP32 mode
        // the standard gemm() auto-detects dtypes and works correctly.
        Tensor q_flat, k_flat, v_flat;
        if (weight_dtype_ == DType::F16) {
            q_flat = gemm_f16f32(normed, L.q, true);
            k_flat = gemm_f16f32(normed, L.k, true);
            v_flat = gemm_f16f32(normed, L.v, true);
        } else {
            q_flat = gemm(normed, L.q, true);
            k_flat = gemm(normed, L.k, true);
            v_flat = gemm(normed, L.v, true);
        }

        q_flat.reshape_inplace({T, Hq_, hd_});
        k_flat.reshape_inplace({T, Hkv_, hd_});
        v_flat.reshape_inplace({T, Hkv_, hd_});

        // ---------------------------------------------------------------
        // 3. RoPE (absolute positions [start_pos, start_pos + T - 1])
        // ---------------------------------------------------------------
        {
            std::vector<float> vc(static_cast<size_t>(T) * half_hd);
            std::vector<float> vs(static_cast<size_t>(T) * half_hd);
            for (int t = 0; t < T; t++) {
                int pos = start_pos + t;
                for (int d = 0; d < half_hd; d++) {
                    float theta = static_cast<float>(pos)
                        / powf(cfg_.rope_theta, 2.0f * d / hd_);
                    size_t idx = static_cast<size_t>(t) * half_hd + d;
                    vc[idx] = cosf(theta);
                    vs[idx] = sinf(theta);
                }
            }
            Tensor cos({T, half_hd}, DType::F32, Device::CUDA);
            Tensor sin({T, half_hd}, DType::F32, Device::CUDA);
            cos.copy_from(vc.data(), cos.nbytes());
            sin.copy_from(vs.data(), sin.nbytes());

            rope(q_flat, &k_flat, cos, sin);
        }
        cudaDeviceSynchronize();

        // ---------------------------------------------------------------
        // 4. Causal self-attention   [T, Hq, hd]
        // ---------------------------------------------------------------
        // ops::attention is causal — it computes softmax(Q*K^T/√d) with
        // a causal mask so token t can only see tokens [0..t].
        Tensor attn_out = attention(q_flat, k_flat, v_flat);
        attn_out.reshape_inplace({T, D_});

        // Cast attention output to match weight dtype before O-proj GEMM.
        // (attention always produces F32; O-proj may expect F16.)
        if (attn_out.dtype() != weight_dtype_) {
            if (weight_dtype_ == DType::F16) {
                attn_out = cast_f32_to_f16(attn_out);
            } else if (weight_dtype_ == DType::F32) {
                attn_out = cast_f16_to_f32(attn_out);
            }
        }

        // ---------------------------------------------------------------
        // 5. O-projection + first residual
        // ---------------------------------------------------------------
        attn_out = gemm(attn_out, L.o, true);
        x = ops::add(x, attn_out);

        // ---------------------------------------------------------------
        // 6. MLP block:  RMSNorm → gate/up GEMM → SiLU → mul → down GEMM
        //    → second residual
        // ---------------------------------------------------------------
        normed = rms_norm(x, L.m_n, cfg_.rms_norm_eps);
        Tensor gate = gemm(normed, L.gate, true);
        Tensor up   = gemm(normed, L.up,   true);
        silu_inplace(gate);
        Tensor mlp = ops::mul(gate, up);
        mlp = gemm(mlp, L.down, true);
        x = ops::add(x, mlp);
    }

    return x;
}

// ============================================================================
// forward_incremental — k tokens through ALL layers with KV cache write
// ============================================================================
//
// Processes k draft tokens one at a time through every transformer layer.
// Each token: RMSNorm → Q/K/V GEMM → RoPE → write_decode_kv →
// paged_attention (with self-attention via write-before-read) →
// O-proj + residual → MLP (SwiGLU) + residual.
//
// Key difference from forward_n_layers():
//   - Writes K/V to the persistent KV cache BEFORE each layer's attention,
//     so paged_attention with attn_seq_len = token_pos+1 includes self-attn.
//   - Uses paged_attention (reads from KV cache) instead of causal self-attn
//
// For simplicity in this first version, tokens are processed ONE AT A TIME
// as decode steps. This avoids the complexity of the batch prefill path
// while still being much faster than re-processing the full prompt.
//
// The caller must ensure blocks are allocated for positions
// [start_pos .. start_pos+k-1] BEFORE calling this method.

Tensor EngineServer::forward_incremental(
    const Tensor& tokens,
    const std::vector<kv_cache::BlockTable>& block_tables,
    int start_pos)
{
    int k = tokens.size(0);
    size_t row_bytes = D_ * dtype_size(weight_dtype_);

    // Allocate output tensor: [k, D_] in weight_dtype_
    Tensor result({k, D_}, weight_dtype_, Device::CUDA);

    for (int i = 0; i < k; i++) {
        // attn_seq_len = token_pos (NOT +1) so that paged_attention
        // only reads positions 0..token_pos-1 and does NOT include
        // self-attention.  This matches step()'s decode behaviour and
        // ensures the K/V cache is consistent between the two paths.
        int token_pos    = start_pos + i;   // absolute RoPE + KV write position
        int attn_seq_len = token_pos;       // attend to positions 0..token_pos-1

        // Extract the i-th token's hidden state from input
        Tensor h_i({1, D_}, weight_dtype_, Device::CUDA);
        cudaMemcpy(h_i.raw(),
                   static_cast<const char*>(tokens.raw()) + i * row_bytes,
                   row_bytes,
                   cudaMemcpyDeviceToDevice);

        // Process through ALL layers with paged_attention + KV cache write.
        // forward_partial handles: norm → Q/K/V → RoPE → write_decode_kv →
        // paged_attention (with self-attention via write-before-read) →
        // O-proj + residual → MLP + residual
        Tensor h_out = forward_partial(h_i, 0, n_layers_,
                                        block_tables,
                                        token_pos, attn_seq_len);

        // Copy output to result tensor
        cudaMemcpy(static_cast<char*>(result.raw()) + i * row_bytes,
                   h_out.raw(),
                   row_bytes,
                   cudaMemcpyDeviceToDevice);
    }

    return result;
}

bool EngineServer::allocate_decode_blocks(RequestState& state) {
    int block_id = kv_allocators_[0]->allocate(0);
    if (block_id < 0) return false;
    for (int l = 1; l < n_layers_; l++) {
        kv_allocators_[l]->allocate(0);
    }
    for (int l = 0; l < n_layers_; l++) {
        state.block_tables[l].append(block_id);
    }
    return true;
}

}  // namespace engine
}  // namespace lightllm
