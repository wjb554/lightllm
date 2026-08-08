/// LightLLM Inference Engine — full generation pipeline with PagedAttention.
///
/// Prefill:   GPU 3-stage or FlashAttention kernel on contiguous Q/K/V, then
///            scatter K/V into paged blocks for decode.
/// Decode:    paged_attention() kernel with online softmax — zero CPU copies
///            of KV cache.  1 token at a time, one thread-block per Q-head.
#include "lightllm/engine/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <random>

#include <cuda_runtime.h>

#include "lightllm/model/model_loader.h"
#include "lightllm/ops/norm.h"
#include "lightllm/ops/rope.h"
#include "lightllm/ops/gemm.h"
#include "lightllm/ops/elementwise.h"
#include "lightllm/ops/sampling.h"
#include "lightllm/ops/lm_head.h"
#include "lightllm/kv_cache/paged_attention.h"

namespace lightllm {
namespace engine {

using namespace model;
using namespace ops;

// =========================================================================
// GPU Prefill Attention Kernels  (from test_generate.cu)
// =========================================================================

// Stage 1: Q·K^T — one block per (token, Q-head)
__global__ void attn_qk_kernel(
    float* scores, const float* q, const float* k,
    int TQ, int Hq, int TK, int Hkv, int D, int groups, float scale)
{
    int t=blockIdx.x, h=blockIdx.y, kvh=h/groups, tid=threadIdx.x;
    for(int s=tid;s<TK;s+=blockDim.x){
        float dot=0;
        for(int d=0;d<D;d++)dot+=q[(t*Hq+h)*D+d]*k[(s*Hkv+kvh)*D+d];
        scores[(t*Hq+h)*TK+s]=dot*scale;
    }
}

// Stage 2: Softmax along last dim
__global__ void attn_softmax_kernel(float* scores,int rows,int cols){
    int row=blockIdx.x,tid=threadIdx.x;
    __shared__ float smem[64];
    int wid=tid/32,lid=tid%32,nw=blockDim.x/32;
    float mx=-1e38f;
    for(int i=tid;i<cols;i+=blockDim.x)mx=fmaxf(mx,scores[row*cols+i]);
    for(int o=16;o>0;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));
    if(lid==0)smem[wid]=mx;__syncthreads();
    if(tid<32){float v=(tid<nw)?smem[tid]:-1e38f;for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,o));if(tid==0)smem[0]=v;}__syncthreads();
    mx=smem[0];float sm=0;
    for(int i=tid;i<cols;i+=blockDim.x){float v=expf(scores[row*cols+i]-mx);scores[row*cols+i]=v;sm+=v;}
    for(int o=16;o>0;o>>=1)sm+=__shfl_down_sync(0xffffffff,sm,o);
    if(lid==0)smem[wid]=sm;__syncthreads();
    if(tid<32){float v=0;for(int w=0;w<nw;w++)v+=smem[w];for(int o=16;o>0;o>>=1)v+=__shfl_down_sync(0xffffffff,v,o);if(tid==0)smem[1]=v;}__syncthreads();
    sm=smem[1];float inv=1.f/(sm+1e-8f);
    for(int i=tid;i<cols;i+=blockDim.x)scores[row*cols+i]*=inv;
}

// Stage 3: S@V — one block per (token, Q-head)
__global__ void attn_sv_kernel(
    float* out, const float* scores, const float* v,
    int TQ,int Hq,int TK,int Hkv,int D,int groups)
{
    int t=blockIdx.x,h=blockIdx.y,kvh=h/groups,tid=threadIdx.x;
    for(int d=tid;d<D;d+=blockDim.x){
        float acc=0;
        for(int s=0;s<TK;s++)
            acc+=scores[(t*Hq+h)*TK+s]*v[(s*Hkv+kvh)*D+d];
        out[(t*Hq+h)*D+d]=acc;
    }
}

// ---- FlashAttention-style tiled prefill kernel (scheme B, long prompts) ----
static constexpr int B_r=64, B_c=64;  // Q tile and KV tile sizes

__global__ void flash_prefill_kernel(
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
static Tensor prefill_attn(const Tensor& q,const Tensor& k,const Tensor& v,
    int P,int Hq,int Hkv,int D,int groups,float scale)
{
    size_t scores_bytes = (size_t)P * Hq * P * sizeof(float);
    bool use_tiled = (scores_bytes > 32ULL * 1024 * 1024);

    if (!use_tiled) {
        // Scheme A: 3-stage (fast for short prompts, simple)
        Tensor scores({P*Hq, P}, DType::F32, Device::CUDA);
        dim3 g1(P, Hq), b1(std::min(256, D));
        attn_qk_kernel<<<g1,b1>>>(scores.data<float>(),q.data<float>(),
            k.data<float>(), P,Hq,P,Hkv,D,groups,scale);
        int rows=P*Hq, blk=std::min(512,((P+31)/32)*32); if(blk<32)blk=32;
        attn_softmax_kernel<<<rows,blk>>>(scores.data<float>(),rows,P);
        Tensor out({P,Hq,D}, DType::F32, Device::CUDA);
        dim3 g3(P, Hq);
        attn_sv_kernel<<<g3,b1>>>(out.data<float>(),scores.data<float>(),
            v.data<float>(), P,Hq,P,Hkv,D,groups);
        return out;
    } else {
        // Scheme B: FlashAttention tiled (memory-efficient for long prompts)
        Tensor out({P,Hq,D}, DType::F32, Device::CUDA);
        int bdim = ((D + 31) / 32) * 32; if (bdim < 32) bdim = 32;
        size_t smem = (B_r * D + 2 * B_c * D) * sizeof(float);
        dim3 grid((P + B_r - 1) / B_r, Hq);
        flash_prefill_kernel<<<grid, bdim, smem>>>(
            out.data<float>(), q.data<float>(), k.data<float>(), v.data<float>(),
            P, Hq, Hkv, D, groups, scale);
        return out;
    }
}

// =========================================================================
// Weight loading helpers
// =========================================================================

static std::vector<float> bf16f32(const std::vector<char>& r){
    std::vector<float> f(r.size()/2);
    for(size_t i=0;i<f.size();i++){uint16_t b=((const uint16_t*)r.data())[i];uint32_t u=b<<16;f[i]=*(float*)&u;}
    return f;
}

static Tensor load_f32(SafetensorsLoader& l,const std::string& n){
    auto* info=l.get_info(n);Tensor cpu=l.load_tensor(n);
    std::vector<char> raw(cpu.nbytes());cpu.copy_to(raw.data(),cpu.nbytes());
    auto f32=bf16f32(raw);Tensor gpu(info->shape,DType::F32,Device::CUDA);
    gpu.copy_from(f32.data(),f32.size()*sizeof(float));return gpu;
}

// =========================================================================
// InferenceEngine
// =========================================================================

InferenceEngine::InferenceEngine(const std::string& model_dir, int max_seq_len) {
    char buf[512];
    snprintf(buf,sizeof(buf),"%s/config.json",model_dir.c_str());
    cfg_ = parse_config(buf);
    snprintf(buf,sizeof(buf),"%s/model.safetensors",model_dir.c_str());
    SafetensorsLoader loader(buf);

    D_=cfg_.hidden_size;Hq_=cfg_.num_attention_heads;
    Hkv_=cfg_.num_key_value_heads;hd_=cfg_.head_dim;
    n_layers_=cfg_.num_hidden_layers;vocab_=cfg_.vocab_size;

    // Determine max_seq_len: use provided override, or config's default
    if (max_seq_len <= 0)
        max_seq_len_ = cfg_.max_position_embeddings;
    else
        max_seq_len_ = max_seq_len;

    // Number of physical blocks = ceiling + extra safety margin
    num_blocks_ = (max_seq_len_ + BLOCK_SIZE - 1) / BLOCK_SIZE + 8;

    printf("Engine: %s, %d layers, %d params, max_seq_len=%d, %d blocks (block_size=%d)\n",
           cfg_.architecture.c_str(),n_layers_,(int)(cfg_.total_params()/1e6),
           max_seq_len_, num_blocks_, BLOCK_SIZE);
    printf("Loading weights...\n");

    embed_w_=load_f32(loader,"model.embed_tokens.weight");
    final_norm_=load_f32(loader,"model.norm.weight");
    for(int i=0;i<n_layers_;i++){
        auto ns=std::to_string(i);
        layers_.push_back(std::make_unique<LayerW>(LayerW{
            load_f32(loader,"model.layers."+ns+".self_attn.q_proj.weight"),
            load_f32(loader,"model.layers."+ns+".self_attn.k_proj.weight"),
            load_f32(loader,"model.layers."+ns+".self_attn.v_proj.weight"),
            load_f32(loader,"model.layers."+ns+".self_attn.o_proj.weight"),
            load_f32(loader,"model.layers."+ns+".input_layernorm.weight"),
            load_f32(loader,"model.layers."+ns+".mlp.gate_proj.weight"),
            load_f32(loader,"model.layers."+ns+".mlp.up_proj.weight"),
            load_f32(loader,"model.layers."+ns+".mlp.down_proj.weight"),
            load_f32(loader,"model.layers."+ns+".post_attention_layernorm.weight"),
        }));
        if(i==0||i==n_layers_-1)printf("  layer %d loaded\n",i);
    }
    printf("All weights loaded (FP32 on GPU)\n");

    // Build one BlockAllocator per layer — each owns its own K/V GPU pool
    printf("Allocating KV cache pools (%d blocks x %d layers)...\n",
           num_blocks_, n_layers_);
    kv_allocators_.reserve(n_layers_);
    for (int l = 0; l < n_layers_; l++) {
        kv_allocators_.push_back(std::make_unique<kv_cache::BlockAllocator>(
            num_blocks_, BLOCK_SIZE, Hkv_, hd_));
    }
    printf("KV cache pools ready (%.1f MB per layer, %.1f MB total)\n",
           num_blocks_ * BLOCK_SIZE * Hkv_ * hd_ * 2 * 4 / 1048576.0,
           num_blocks_ * BLOCK_SIZE * Hkv_ * hd_ * 2 * 4 * n_layers_ / 1048576.0);
}

// ---- Block helpers: operate on ALL layers in lockstep ----

int InferenceEngine::allocate_block(uint64_t token_hash) {
    if (kv_allocators_.empty()) return -1;
    int block_id = kv_allocators_[0]->allocate(token_hash);
    if (block_id < 0) return -1;
    for (int l = 1; l < n_layers_; l++) {
        kv_allocators_[l]->allocate(token_hash);
    }
    return block_id;
}

void InferenceEngine::release_block(int block_id) {
    for (auto& alloc : kv_allocators_) {
        alloc->release(block_id);
    }
}

// ---- Scatter prefill K/V into paged blocks ----

void InferenceEngine::scatter_prefill_kv(
    int layer, const Tensor& k_contig, const Tensor& v_contig,
    int n_tokens, const kv_cache::BlockTable& bt)
{
    const float* k_src = k_contig.data<float>();
    const float* v_src = v_contig.data<float>();
    size_t row_bytes = static_cast<size_t>(Hkv_) * hd_ * sizeof(float);
    auto& alloc = *kv_allocators_[layer];

    for (int pos = 0; pos < n_tokens; pos++) {
        int blk_idx = pos / BLOCK_SIZE;
        int offset  = pos % BLOCK_SIZE;
        int phys    = bt[blk_idx];

        void* dst_k = static_cast<char*>(alloc.k_data(phys))
                      + offset * row_bytes;
        void* dst_v = static_cast<char*>(alloc.v_data(phys))
                      + offset * row_bytes;
        const void* src_k = k_src + pos * Hkv_ * hd_;
        const void* src_v = v_src + pos * Hkv_ * hd_;

        cudaMemcpy(dst_k, src_k, row_bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst_v, src_v, row_bytes, cudaMemcpyDeviceToDevice);
    }
}

// ---- Write single-token decode K/V into paged block ----

void InferenceEngine::write_decode_kv(
    int layer, int token_pos, const Tensor& k_new, const Tensor& v_new,
    const kv_cache::BlockTable& bt)
{
    int blk_idx = token_pos / BLOCK_SIZE;
    int offset  = token_pos % BLOCK_SIZE;
    int phys    = bt[blk_idx];
    size_t row_bytes = static_cast<size_t>(Hkv_) * hd_ * sizeof(float);
    auto& alloc = *kv_allocators_[layer];

    void* dst_k = static_cast<char*>(alloc.k_data(phys))
                  + offset * row_bytes;
    void* dst_v = static_cast<char*>(alloc.v_data(phys))
                  + offset * row_bytes;

    cudaMemcpy(dst_k, k_new.raw(), row_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst_v, v_new.raw(), row_bytes, cudaMemcpyDeviceToDevice);
}

// =========================================================================
// generate() — full prefill + decode with PagedAttention
// =========================================================================

GenerateResult InferenceEngine::generate(
    const std::vector<int>& prompt_ids, const GenerateParams& params)
{
    GenerateResult result;
    int P          = static_cast<int>(prompt_ids.size());
    int max_seq    = P + params.max_new_tokens;
    int groups     = Hq_ / Hkv_;
    float attn_scale = 1.0f / sqrtf(float(hd_));

    // -----------------------------------------------------------------
    // 0. SETUP — per-request block table + GPU-side helpers
    // -----------------------------------------------------------------
    int max_blocks_per_seq = (max_seq_len_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kv_cache::BlockTable bt(max_blocks_per_seq);

    // GPU-side block table [1, max_blocks_per_seq], init to -1
    Tensor d_block_table({1, max_blocks_per_seq}, DType::I32, Device::CUDA);
    {
        std::vector<int> bt_init(max_blocks_per_seq, -1);
        d_block_table.copy_from(bt_init.data(), bt_init.size() * sizeof(int));
    }

    // GPU-side sequence length (single int)
    Tensor d_seq_len({1}, DType::I32, Device::CUDA);

    auto t0 = std::chrono::steady_clock::now();

    // -----------------------------------------------------------------
    // 1. EMBEDDING — lookup from embed_w_ on CPU, upload to GPU
    // -----------------------------------------------------------------
    Tensor h({P, D_}, DType::F32, Device::CUDA);
    {
        auto ew_cpu = std::vector<float>(embed_w_.numel());
        embed_w_.copy_to(ew_cpu.data(), embed_w_.nbytes());
        std::vector<float> emb(P * D_);
        for (int t = 0; t < P; t++) {
            int tid = prompt_ids[t];
            if (tid < 0 || tid >= vocab_) tid = 0;
            memcpy(emb.data() + t * D_, ew_cpu.data() + tid * D_,
                   D_ * sizeof(float));
        }
        h.copy_from(emb.data(), emb.size() * sizeof(float));
    }

    // -----------------------------------------------------------------
    // 2. PREFILL — process all P prompt tokens through all layers
    // -----------------------------------------------------------------
    for (int l = 0; l < n_layers_; l++) {
        auto& L = *layers_[l];

        // --- Q/K/V projection + RoPE ---
        auto normed = rms_norm(h, L.a_n, cfg_.rms_norm_eps);
        auto q = gemm(normed, L.q, true);
        auto k = gemm(normed, L.k, true);
        auto v = gemm(normed, L.v, true);
        auto q3 = q.view({P, Hq_, hd_});
        auto k3 = k.view({P, Hkv_, hd_});
        auto v3 = v.view({P, Hkv_, hd_});

        // RoPE for prefill positions 0..P-1
        Tensor cos({P, hd_ / 2}, DType::F32, Device::CUDA);
        Tensor sin({P, hd_ / 2}, DType::F32, Device::CUDA);
        {
            std::vector<float> vc(P * hd_ / 2), vs(P * hd_ / 2);
            for (int t = 0; t < P; t++)
                for (int d = 0; d < hd_ / 2; d++) {
                    float theta = float(t) / powf(cfg_.rope_theta, 2.f * d / hd_);
                    vc[t * hd_ / 2 + d] = cosf(theta);
                    vs[t * hd_ / 2 + d] = sinf(theta);
                }
            cos.copy_from(vc.data(), cos.nbytes());
            sin.copy_from(vs.data(), sin.nbytes());
        }
        rope(q3, &k3, cos, sin);
        cudaDeviceSynchronize();

        // --- Allocate blocks for prefill K/V (first layer drives allocation) ---
        int n_blocks_needed = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (l == 0) {
            for (int b = 0; b < n_blocks_needed; b++) {
                int block_id = allocate_block(/*token_hash=*/0);
                if (block_id < 0) {
                    fprintf(stderr, "FATAL: KV cache pool exhausted during prefill!\n");
                    return result;
                }
                bt.append(block_id);
            }
        }

        // --- Scatter K/V into paged blocks ---
        scatter_prefill_kv(l, k3, v3, P, bt);

        // --- GPU prefill attention on contiguous Q/K/V ---
        Tensor attn_out = prefill_attn(q3, k3, v3, P, Hq_, Hkv_, hd_,
                                       groups, attn_scale);

        // --- O-projection + residual ---
        attn_out.reshape_inplace({P, D_});
        attn_out = gemm(attn_out, L.o, true);
        h = ops::add(h, attn_out);

        // --- MLP ---
        normed = rms_norm(h, L.m_n, cfg_.rms_norm_eps);
        auto gate = gemm(normed, L.gate, true);
        auto up   = gemm(normed, L.up, true);
        silu_inplace(gate);
        auto mlp = ops::mul(gate, up);
        mlp = gemm(mlp, L.down, true);
        h = ops::add(h, mlp);
    }

    // --- Final norm + extract last hidden state ---
    h = rms_norm(h, final_norm_, cfg_.rms_norm_eps);
    Tensor last_h({1, D_}, DType::F32, Device::CUDA);
    cudaMemcpy(last_h.raw(), static_cast<const float*>(h.raw()) + (P - 1) * D_,
               D_ * sizeof(float), cudaMemcpyDeviceToDevice);

    result.token_ids = prompt_ids;

    // -----------------------------------------------------------------
    // 3. DECODE LOOP — autoregressive generation
    // -----------------------------------------------------------------
    int cur_len = P;
    std::mt19937 rng(params.seed);

    for (int step = 0; step < params.max_new_tokens && cur_len < max_seq; step++) {

        // --- Maybe allocate a fresh block ---
        if (cur_len % BLOCK_SIZE == 0) {
            int block_id = allocate_block(/*token_hash=*/0);
            if (block_id < 0) {
                printf("KV cache pool exhausted at token %d\n", cur_len);
                break;
            }
            bt.append(block_id);
        }

        // --- Update GPU-side metadata ---
        int seq_len_val = cur_len + 1;  // all historical + current token
        d_seq_len.copy_from(&seq_len_val, sizeof(int));

        // Copy block table to GPU
        {
            std::vector<int> bt_cpu(max_blocks_per_seq, -1);
            for (int i = 0; i < bt.size(); i++) bt_cpu[i] = bt[i];
            d_block_table.copy_from(bt_cpu.data(),
                                    bt_cpu.size() * sizeof(int));
        }

        // --- Hidden state for this iteration ---
        Tensor h_dec({1, D_}, DType::F32, Device::CUDA);
        cudaMemcpy(h_dec.raw(), last_h.raw(), D_ * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        for (int l = 0; l < n_layers_; l++) {
            auto& L = *layers_[l];

            // --- Q/K/V projection + RoPE ---
            auto normed = rms_norm(h_dec, L.a_n, cfg_.rms_norm_eps);
            auto q = gemm(normed, L.q, true);
            auto k = gemm(normed, L.k, true);
            auto v = gemm(normed, L.v, true);
            auto q3 = q.view({1, Hq_, hd_});
            auto k3 = k.view({1, Hkv_, hd_});
            auto v3 = v.view({1, Hkv_, hd_});

            // RoPE for current position
            Tensor cos({1, hd_ / 2}, DType::F32, Device::CUDA);
            Tensor sin({1, hd_ / 2}, DType::F32, Device::CUDA);
            {
                std::vector<float> vc(hd_ / 2), vs(hd_ / 2);
                for (int d = 0; d < hd_ / 2; d++) {
                    float theta = float(cur_len) / powf(cfg_.rope_theta, 2.f * d / hd_);
                    vc[d] = cosf(theta);
                    vs[d] = sinf(theta);
                }
                cos.copy_from(vc.data(), cos.nbytes());
                sin.copy_from(vs.data(), sin.nbytes());
            }
            rope(q3, &k3, cos, sin);
            cudaDeviceSynchronize();

            // --- Write single-token K/V into paged block ---
            write_decode_kv(l, cur_len, k3, v3, bt);

            // --- PagedAttention: GPU kernel, zero CPU copies! ---
            Tensor attn_out = kv_cache::paged_attention(
                q3,
                d_block_table.data<int>(),
                max_blocks_per_seq,
                d_seq_len.data<int>(),
                *kv_allocators_[l]);

            // --- O-projection + residual ---
            attn_out.reshape_inplace({1, D_});
            attn_out = gemm(attn_out, L.o, true);
            h_dec = ops::add(h_dec, attn_out);

            // --- MLP ---
            normed = rms_norm(h_dec, L.m_n, cfg_.rms_norm_eps);
            auto gate = gemm(normed, L.gate, true);
            auto up   = gemm(normed, L.up, true);
            silu_inplace(gate);
            auto mlp = ops::mul(gate, up);
            mlp = gemm(mlp, L.down, true);
            h_dec = ops::add(h_dec, mlp);
        }

        // --- Final norm + lm_head + sampling ---
        h_dec = rms_norm(h_dec, final_norm_, cfg_.rms_norm_eps);

        auto logits_gpu = ops::lm_head_logits(h_dec, embed_w_);
        std::vector<float> logits(vocab_);
        logits_gpu.copy_to(logits.data(), vocab_ * sizeof(float));

        ops::SamplingParams sp;
        sp.temperature = params.temperature;
        sp.top_k = params.top_k;
        sp.top_p = params.top_p;
        sp.min_p = 0.0f;
        sp.seed = params.seed + step;
        int next = ops::sample(logits.data(), vocab_, sp, rng);

        result.token_ids.push_back(next);
        cur_len++;

        last_h = std::move(h_dec);

        if (next == params.eos_token_id) break;
    }

    // ---- Cleanup: release all allocated blocks ----
    for (int i = 0; i < bt.size(); i++) {
        release_block(bt[i]);
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    int new_tokens = static_cast<int>(result.token_ids.size()) - P;
    printf("Generated %d tokens in %lld ms (%.1f tok/s)\n",
           new_tokens, elapsed, new_tokens * 1000.0 / elapsed);

    return result;
}

// =========================================================================
// Tokenizer stubs (unchanged)
// =========================================================================

std::vector<int> InferenceEngine::tokenize(const std::string&) const {
    return {};
}

std::string InferenceEngine::detokenize(const std::vector<int>&) const {
    return {};
}

}  // namespace engine
}  // namespace lightllm
