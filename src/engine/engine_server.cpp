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
#include "lightllm/ops/lm_head.h"
#include "lightllm/tokenizer/tokenizer.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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
// GPU Prefill Attention Kernels  (copied from engine_paged.cpp)
// ============================================================================

// Stage 1: Q*K^T — one block per (token, Q-head)
__global__ void es_attn_qk_kernel(
    float* scores, const float* q, const float* k,
    int TQ, int Hq, int TK, int Hkv, int D, int groups, float scale)
{
    int t=blockIdx.x, h=blockIdx.y, kvh=h/groups, tid=threadIdx.x;
    for (int s=tid; s<TK; s+=blockDim.x) {
        float dot=0;
        for (int d=0; d<D; d++) dot+=q[(t*Hq+h)*D+d]*k[(s*Hkv+kvh)*D+d];
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
__global__ void es_attn_sv_kernel(
    float* out, const float* scores, const float* v,
    int TQ, int Hq, int TK, int Hkv, int D, int groups)
{
    int t=blockIdx.x, h=blockIdx.y, kvh=h/groups, tid=threadIdx.x;
    for (int d=tid; d<D; d+=blockDim.x) {
        float acc=0;
        for (int s=0; s<TK; s++)
            acc+=scores[(t*Hq+h)*TK+s]*v[(s*Hkv+kvh)*D+d];
        out[(t*Hq+h)*D+d]=acc;
    }
}

// FlashAttention-style tiled prefill kernel
static constexpr int B_r=64, B_c=64;

__global__ void es_flash_prefill_kernel(
    float* __restrict__ out, const float* __restrict__ q,
    const float* __restrict__ k, const float* __restrict__ v,
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
        q_tile[i] = (qi < P) ? q[(qi * Hq + q_head) * D + d] : 0.0f;
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
            k_tile[i] = valid ? k[(ki * Hkv + kv_head) * D + d] : 0.0f;
            v_tile[i] = valid ? v[(ki * Hkv + kv_head) * D + d] : 0.0f;
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
                = acc[qi] / (l_val[qi] + 1e-8f);
    }
}

// Adaptive prefill attention: short prompts -> 3-stage, long prompts -> tiled
static Tensor prefill_attn(const Tensor& q, const Tensor& k, const Tensor& v,
    int P, int Hq, int Hkv, int D, int groups, float scale)
{
    size_t scores_bytes = (size_t)P * Hq * P * sizeof(float);
    bool use_tiled = (scores_bytes > 32ULL * 1024 * 1024);

    if (!use_tiled) {
        // Scheme A: 3-stage
        Tensor scores({P * Hq, P}, DType::F32, Device::CUDA);
        dim3 g1(P, Hq), b1(std::min(256, D));
        es_attn_qk_kernel<<<g1,b1>>>(scores.data<float>(), q.data<float>(),
            k.data<float>(), P, Hq, P, Hkv, D, groups, scale);
        int rows=P*Hq, blk=std::min(512, ((P+31)/32)*32);
        if (blk<32) blk=32;
        es_attn_softmax_kernel<<<rows,blk>>>(scores.data<float>(), rows, P);
        Tensor out({P, Hq, D}, DType::F32, Device::CUDA);
        dim3 g3(P, Hq);
        es_attn_sv_kernel<<<g3,b1>>>(out.data<float>(), scores.data<float>(),
            v.data<float>(), P, Hq, P, Hkv, D, groups);
        return out;
    } else {
        // Scheme B: FlashAttention tiled
        Tensor out({P, Hq, D}, DType::F32, Device::CUDA);
        int bdim = ((D + 31) / 32) * 32; if (bdim < 32) bdim = 32;
        size_t smem = (B_r * D + 2 * B_c * D) * sizeof(float);
        dim3 grid((P + B_r - 1) / B_r, Hq);
        es_flash_prefill_kernel<<<grid, bdim, smem>>>(
            out.data<float>(), q.data<float>(), k.data<float>(), v.data<float>(),
            P, Hq, Hkv, D, groups, scale);
        return out;
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
                           kv_cache::PrefixCachePolicy prefix_cache_policy)
    : max_batch_tokens_(max_batch_tokens)
    , kv_cache_mb_(kv_cache_mb)
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

    embed_w_    = load_f32(loader, "model.embed_tokens.weight");
    final_norm_ = load_f32(loader, "model.norm.weight");
    for (int i = 0; i < n_layers_; i++) {
        auto ns = std::to_string(i);
        layers_.push_back(std::make_unique<EngineLayerW>(EngineLayerW{
            load_f32(loader, "model.layers."+ns+".self_attn.q_proj.weight"),
            load_f32(loader, "model.layers."+ns+".self_attn.k_proj.weight"),
            load_f32(loader, "model.layers."+ns+".self_attn.v_proj.weight"),
            load_f32(loader, "model.layers."+ns+".self_attn.o_proj.weight"),
            load_f32(loader, "model.layers."+ns+".input_layernorm.weight"),
            load_f32(loader, "model.layers."+ns+".mlp.gate_proj.weight"),
            load_f32(loader, "model.layers."+ns+".mlp.up_proj.weight"),
            load_f32(loader, "model.layers."+ns+".mlp.down_proj.weight"),
            load_f32(loader, "model.layers."+ns+".post_attention_layernorm.weight"),
        }));
        if (i == 0 || i == n_layers_ - 1)
            printf("  layer %d loaded\n", i);
    }
    printf("All weights loaded (FP32 on GPU)\n");

    // Allocate KV cache pools
    printf("Allocating KV cache pools (%d blocks x %d layers)...\n",
           num_blocks_, n_layers_);
    kv_allocators_.reserve(n_layers_);
    for (int l = 0; l < n_layers_; l++) {
        kv_allocators_.push_back(std::make_unique<kv_cache::BlockAllocator>(
            num_blocks_, BLOCK_SIZE, Hkv_, hd_, prefix_cache_policy_));
    }
    printf("KV cache pools ready (%.1f MB per layer, %.1f MB total)\n",
           num_blocks_ * BLOCK_SIZE * Hkv_ * hd_ * 2 * 4 / 1048576.0,
           num_blocks_ * BLOCK_SIZE * Hkv_ * hd_ * 2 * 4 * n_layers_ / 1048576.0);

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

    // Allocate next_hidden_state (uninitialised until first prefill)
    state->next_hidden_state = Tensor({1, D}, DType::F32, Device::CUDA);

    return state;
}

// ============================================================================
// release_request — free ALL blocks across ALL layers
// ============================================================================

void EngineServer::release_request(RequestState& state) {
    if (state.block_tables.empty()) return;
    auto& bt0 = state.block_tables[0];
    for (int i = 0; i < bt0.size(); i++) {
        int block_id = bt0[i];
        for (auto& alloc : kv_allocators_) {
            alloc->release(block_id);
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
    const std::vector<kv_cache::BlockTable>& block_tables)
{
    int blk_idx = token_pos / BLOCK_SIZE;
    int offset  = token_pos % BLOCK_SIZE;
    int phys    = block_tables[layer][blk_idx];
    size_t row_bytes = static_cast<size_t>(Hkv_) * hd_ * sizeof(float);
    auto& alloc = *kv_allocators_[layer];

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
    Tensor h({total_tokens, D_}, DType::F32, Device::CUDA);
    entry_map.clear();
    entry_map.reserve(step.entries.size());

    // Copy embed weight to CPU once for all prefill lookups
    std::vector<float> ew_cpu(embed_w_.numel());
    embed_w_.copy_to(ew_cpu.data(), embed_w_.nbytes());

    int row = 0;
    for (const auto& entry : step.entries) {
        int start = row;
        auto it = states.find(entry.request_idx);
        if (it == states.end()) {
            fprintf(stderr, "FATAL: RequestState not found for id=%d\n",
                    entry.request_idx);
            row += entry.num_tokens;
            entry_map.push_back({start, row});
            continue;
        }
        auto& state = *it->second;

        if (entry.is_prefill) {
            // Embed prompt tokens [start_pos, start_pos + num_tokens)
            int chunk_size = entry.num_tokens;
            int chunk_start = entry.start_pos;
            std::vector<float> emb(chunk_size * D_);
            for (int t = 0; t < chunk_size; t++) {
                int tid = state.prompt_tokens[chunk_start + t];
                if (tid < 0 || tid >= vocab_) tid = 0;
                memcpy(emb.data() + t * D_,
                       ew_cpu.data() + tid * D_,
                       D_ * sizeof(float));
            }
            cudaMemcpy(static_cast<float*>(h.raw()) + row * D_,
                       emb.data(),
                       chunk_size * D_ * sizeof(float),
                       cudaMemcpyHostToDevice);
            row += chunk_size;
        } else {
            // Decode: copy next_hidden_state from previous step
            if (!state.next_hidden_state.defined()) {
                fprintf(stderr, "FATAL: decode entry %d has no hidden state\n",
                        entry.request_idx);
                // Zero-fill as fallback
                std::vector<float> zeros(D_, 0.0f);
                cudaMemcpy(static_cast<float*>(h.raw()) + row * D_,
                           zeros.data(), D_ * sizeof(float),
                           cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy(static_cast<float*>(h.raw()) + row * D_,
                           state.next_hidden_state.raw(),
                           D_ * sizeof(float),
                           cudaMemcpyDeviceToDevice);
            }
            row += 1;
        }

        entry_map.push_back({start, row});
    }

    return h;
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
        Tensor q_flat = gemm(normed, L.q, true);  // [T, Hq*hd]
        Tensor k_flat = gemm(normed, L.k, true);  // [T, Hkv*hd]
        Tensor v_flat = gemm(normed, L.v, true);  // [T, Hkv*hd]

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
                for (int t = 0; t < entry.num_tokens; t++) {
                    int pos = entry.start_pos + t;
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
            kv_cache::scatter_prefill_kv_batched_gpu(
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
            kv_cache::first_prefill_attn_batched_gpu(
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
                                    state.block_tables);
                }
            }
        }

        // --- 3e. O-projection + residual (batched) ---
        Tensor attn_out_2d = attn_flat.view({total_tokens, D_});
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
    h = rms_norm(h, final_norm_, cfg_.rms_norm_eps);

    // Prefill entries: update state (no output tokens)
    for (int idx : prefill_indices) {
        const auto& entry = step.entries[idx];
        auto [start, end] = entry_map[idx];
        auto it = states.find(entry.request_idx);
        if (it == states.end()) continue;
        auto& state = *it->second;

        // Store the last token's hidden state for the first decode step
        Tensor last_h({1, D_}, DType::F32, Device::CUDA);
        cudaMemcpy(last_h.raw(),
                   static_cast<const float*>(h.raw()) + (end - 1) * D_,
                   D_ * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        state.next_hidden_state = std::move(last_h);
        state.num_prefilled += entry.num_tokens;
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
        Tensor tok_h({1, D_}, DType::F32, Device::CUDA);
        cudaMemcpy(tok_h.raw(),
                   static_cast<const float*>(h.raw()) + start * D_,
                   D_ * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        // LM head -> logits on GPU
        Tensor logits_gpu = ops::lm_head_logits(tok_h, embed_w_);
        std::vector<float> logits_cpu(vocab_);
        logits_gpu.copy_to(logits_cpu.data(), vocab_ * sizeof(float));

        // Sample — use greedy when grammar is active (the grammar
        // already constrains valid tokens; randomness adds noise).
        SamplingParams sp;
        sp.temperature = state.grammar_matcher ? 0.0f : 1.0f;
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
        state.next_hidden_state = std::move(tok_h);

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

    return results;
}

}  // namespace engine
}  // namespace lightllm
