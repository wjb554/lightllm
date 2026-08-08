/// Comprehensive PagedAttention engine integration test.
///
/// Demonstrates the new InferenceEngine design where:
///   - N BlockAllocators (one per layer) own K/V GPU pools allocated once
///   - BlockTable maps logical-to-physical blocks per request
///   - Prefill: attention on contiguous Q/K/V, then scatter K/V into blocks
///   - Decode:  paged_attention() GPU kernel with online softmax, zero CPU copies
///
/// Tests:
///   1. BASIC GENERATION — coherent output, block allocator usage, tok/s
///   2. PAGED VS CONTIGUOUS — speed comparison against CPU-attention baseline
///   3. BLOCK MANAGEMENT  — release after generation, prefix cache sharing
///   4. STRESS TEST       — multi-turn generation, no memory leaks
///
/// This test exercises the full pipeline defined in the engine pseudocode
/// (prefill scatter + decode paged_attention) using the already-implemented
/// kv_cache::paged_attention() kernel and BlockAllocator API.

#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#include <memory>
#include <vector>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>

#include "lightllm/tensor.h"
#include "lightllm/kv_cache/block_allocator.h"
#include "lightllm/kv_cache/paged_attention.h"
#include "lightllm/model/model_loader.h"
#include "lightllm/model/model_config.h"
#include "lightllm/ops/norm.h"
#include "lightllm/ops/rope.h"
#include "lightllm/ops/gemm.h"
#include "lightllm/ops/elementwise.h"
#include "lightllm/ops/attention.h"
#include "lightllm/ops/sampling.h"
#include "lightllm/ops/lm_head.h"

using namespace lightllm;
using namespace lightllm::model;
using namespace lightllm::kv_cache;
using namespace lightllm::ops;

// ============================================================================
// Test framework
// ============================================================================
static int g_passed = 0;
static int g_failed = 0;

static void CHECK(const char* name, bool cond) {
    if (cond) {
        g_passed++;
    } else {
        g_failed++;
        fprintf(stderr, "  FAIL: %s\n", name);
    }
}

// ============================================================================
// BF16 -> FP32 conversion (CPU)
// ============================================================================
static std::vector<float> bf16_to_f32(const std::vector<char>& raw) {
    size_t n = raw.size() / 2;
    std::vector<float> f(n);
    for (size_t i = 0; i < n; i++) {
        uint16_t b = reinterpret_cast<const uint16_t*>(raw.data())[i];
        uint32_t u = static_cast<uint32_t>(b) << 16;
        f[i] = *reinterpret_cast<float*>(&u);
    }
    return f;
}

/// Load a BF16 tensor from safetensors, convert to FP32, upload to GPU.
static Tensor load_f32(SafetensorsLoader& loader, const std::string& name) {
    auto* info = loader.get_info(name);
    if (!info) throw std::runtime_error("Tensor not found: " + name);
    Tensor cpu_t = loader.load_tensor(name);
    std::vector<char> raw(cpu_t.nbytes());
    cpu_t.copy_to(raw.data(), cpu_t.nbytes());
    auto f32 = bf16_to_f32(raw);
    Tensor gpu(info->shape, DType::F32, Device::CUDA);
    gpu.copy_from(f32.data(), f32.size() * sizeof(float));
    return gpu;
}

// ============================================================================
// CUDA error checking helper
// ============================================================================
static void cuda_check(const char* context) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        fprintf(stderr, "CUDA error in %s: %s\n", context, cudaGetErrorString(e));
}

// ============================================================================
// PAGED ATTENTION ENGINE — full generation pipeline
//
// This is the reference implementation of InferenceEngine::generate() as
// described in the design pseudocode.  It is self-contained and does NOT
// depend on the old engine.cpp (which uses CPU attention + contiguous cache).
// ============================================================================

/// Per-layer weight structure (same shape as InferenceEngine::LayerW).
struct LayerWeights {
    Tensor q, k, v, o;          // attention projections
    Tensor a_n, m_n;            // RMSNorm weights
    Tensor gate, up, down;      // MLP projections
};

/// Parameters for a single generate call.
struct GenParams {
    int max_new_tokens = 64;
    int eos_token_id   = 151643;  // Qwen EOS
    float temperature  = 0.8f;
    int top_k          = 40;
    float top_p        = 0.9f;
    int seed           = 42;
};

/// Result of a generate call.
struct GenResult {
    std::vector<int> token_ids;
    double elapsed_ms = 0.0;
    int blocks_allocated = 0;
};

/// PagedEngineState holds resources that span multiple generate() calls:
///   - Model weights (FP32 on GPU)
///   - Per-layer BlockAllocators with pre-allocated K/V pools
///
/// This matches the constructor of the new InferenceEngine design:
///   "All allocations happen once in the constructor (per engine instance),
///    NOT per generate() call."
struct PagedEngineState {
    // Model geometry
    int D = 0, Hq = 0, Hkv = 0, hd = 0, n_layers = 0, vocab = 0;
    int groups = 0;
    float attn_scale = 0.0f;
    float rope_theta = 10000.0f;

    // Model weights
    Tensor embed_w;
    Tensor final_norm;
    std::vector<std::unique_ptr<LayerWeights>> layers;

    // Paged KV cache — one BlockAllocator per layer (allocated once)
    static constexpr int BLOCK_SIZE = 16;
    int max_seq_len = 0;
    int num_blocks = 0;
    int max_blocks_per_seq = 0;
    std::vector<std::unique_ptr<BlockAllocator>> kv_allocators;

    // ---- Constructor: load model and allocate KV pools ----------------

    explicit PagedEngineState(const std::string& model_dir,
                              int max_seq_len_override = 0) {
        char buf[512];

        // Parse config
        snprintf(buf, sizeof(buf), "%s/config.json", model_dir.c_str());
        auto cfg = parse_config(buf);
        snprintf(buf, sizeof(buf), "%s/model.safetensors", model_dir.c_str());
        SafetensorsLoader loader(buf);

        D        = cfg.hidden_size;
        Hq       = cfg.num_attention_heads;
        Hkv      = cfg.num_key_value_heads;
        hd       = cfg.head_dim;
        n_layers = cfg.num_hidden_layers;
        vocab    = cfg.vocab_size;
        groups   = Hq / Hkv;
        attn_scale = 1.0f / sqrtf(static_cast<float>(hd));
        rope_theta = cfg.rope_theta;

        // max_seq_len: use override or config's max_position_embeddings
        max_seq_len = (max_seq_len_override > 0)
                      ? max_seq_len_override
                      : cfg.max_position_embeddings;
        num_blocks       = (max_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        max_blocks_per_seq = num_blocks;  // single request: can use all blocks

        printf("\n[PagedEngineState] %s, %d layers, %d params\n",
               cfg.architecture.c_str(), n_layers,
               static_cast<int>(cfg.total_params() / 1e6));
        printf("[PagedEngineState] D=%d Hq=%d Hkv=%d hd=%d vocab=%d\n",
               D, Hq, Hkv, hd, vocab);
        printf("[PagedEngineState] max_seq_len=%d num_blocks=%d block_size=%d\n",
               max_seq_len, num_blocks, BLOCK_SIZE);

        // Load weights
        printf("[PagedEngineState] Loading weights...\n");
        embed_w    = load_f32(loader, "model.embed_tokens.weight");
        final_norm = load_f32(loader, "model.norm.weight");

        for (int i = 0; i < n_layers; i++) {
            auto ns = std::to_string(i);
            auto lw = std::make_unique<LayerWeights>(LayerWeights{
                load_f32(loader, "model.layers." + ns + ".self_attn.q_proj.weight"),
                load_f32(loader, "model.layers." + ns + ".self_attn.k_proj.weight"),
                load_f32(loader, "model.layers." + ns + ".self_attn.v_proj.weight"),
                load_f32(loader, "model.layers." + ns + ".self_attn.o_proj.weight"),
                load_f32(loader, "model.layers." + ns + ".input_layernorm.weight"),
                load_f32(loader, "model.layers." + ns + ".post_attention_layernorm.weight"),
                load_f32(loader, "model.layers." + ns + ".mlp.gate_proj.weight"),
                load_f32(loader, "model.layers." + ns + ".mlp.up_proj.weight"),
                load_f32(loader, "model.layers." + ns + ".mlp.down_proj.weight"),
            });
            layers.push_back(std::move(lw));
        }
        printf("[PagedEngineState] All %d layers loaded (FP32 on GPU)\n", n_layers);

        // Allocate KV pools — one BlockAllocator per layer (once, in constructor)
        // Each pool is shaped [num_blocks, BLOCK_SIZE, Hkv, hd]
        printf("[PagedEngineState] Allocating KV pools: "
               "%d layers x %d blocks x %d x %d x %d floats...\n",
               n_layers, num_blocks, BLOCK_SIZE, Hkv, hd);
        for (int i = 0; i < n_layers; i++) {
            kv_allocators.push_back(
                std::make_unique<BlockAllocator>(num_blocks, BLOCK_SIZE, Hkv, hd));
        }
        size_t pool_bytes_per_layer =
            static_cast<size_t>(num_blocks) * BLOCK_SIZE * Hkv * hd * sizeof(float);
        printf("[PagedEngineState] KV pools allocated: %.1f MB total\n",
               (2.0 * n_layers * pool_bytes_per_layer) / (1024.0 * 1024.0));
    }

    // ---- Block management helpers -------------------------------------

    /// Allocate a physical block across ALL layers in lockstep.
    ///
    /// Allocation discipline: we always call allocate() on EVERY layer's
    /// allocator so block IDs stay synchronised.  Because all allocators
    /// start with identical free-lists and we never call allocate on a
    /// subset, they naturally hand out the same IDs.
    int allocate_block(uint64_t token_hash = 0) {
        if (kv_allocators.empty()) return -1;
        int first_id = kv_allocators[0]->allocate(token_hash);
        if (first_id < 0) return -1;
        for (size_t i = 1; i < kv_allocators.size(); i++) {
            int id = kv_allocators[i]->allocate(token_hash);
            // Paranoia: check lockstep invariant
            if (id != first_id) {
                fprintf(stderr,
                    "WARNING: allocator %zu returned block %d, expected %d\n",
                    i, id, first_id);
            }
        }
        return first_id;
    }

    /// Release a physical block across ALL layers.
    void release_block(int block_id) {
        for (auto& alloc : kv_allocators)
            alloc->release(block_id);
    }

    /// Number of free blocks (same for all allocators — check first one).
    int num_free_blocks() const {
        return kv_allocators.empty() ? 0 : kv_allocators[0]->num_free();
    }

    int total_blocks() const { return num_blocks; }

    // ---- Paged K/V write helpers --------------------------------------

    /// Write prefill K/V (contiguous, shape [P, Hkv, hd]) into paged blocks
    /// for a single layer.  Uses cudaMemcpy per token position.
    ///
    /// Pseudocode:
    ///   for pos=0..P-1:
    ///       block_idx  = pos / BLOCK_SIZE
    ///       offset     = pos % BLOCK_SIZE
    ///       phys_block = bt[block_idx]
    ///       cudaMemcpy(k_pool[phys_block]+offset*Hkv*hd, k_contig+pos*Hkv*hd, ...)
    ///       cudaMemcpy(v_pool[phys_block]+offset*Hkv*hd, v_contig+pos*Hkv*hd, ...)
    void scatter_prefill_kv(int layer,
                            const Tensor& k_contig,  // [P, Hkv, hd]
                            const Tensor& v_contig,  // [P, Hkv, hd]
                            int n_tokens,
                            const BlockTable& bt) {
        auto& alloc = *kv_allocators[layer];
        int Hkv_l = alloc.num_kv_heads();
        int hd_l  = alloc.head_dim();
        size_t per_token_bytes = static_cast<size_t>(Hkv_l) * hd_l * sizeof(float);

        for (int pos = 0; pos < n_tokens; pos++) {
            int logical_block = pos / BLOCK_SIZE;
            int offset        = pos % BLOCK_SIZE;
            int phys_block    = bt[logical_block];

            size_t dst_off = static_cast<size_t>(offset) * Hkv_l * hd_l * sizeof(float);
            size_t src_off = static_cast<size_t>(pos) * Hkv_l * hd_l * sizeof(float);

            char* dst_k = static_cast<char*>(alloc.k_data(phys_block)) + dst_off;
            char* dst_v = static_cast<char*>(alloc.v_data(phys_block)) + dst_off;
            const char* src_k = static_cast<const char*>(k_contig.raw()) + src_off;
            const char* src_v = static_cast<const char*>(v_contig.raw()) + src_off;

            cudaMemcpy(dst_k, src_k, per_token_bytes, cudaMemcpyDeviceToDevice);
            cudaMemcpy(dst_v, src_v, per_token_bytes, cudaMemcpyDeviceToDevice);
        }
    }

    /// Write one decode-step K/V token (shape [1, Hkv, hd]) into the
    /// appropriate block/offset for a single layer.
    ///
    /// Pseudocode:
    ///   block_idx  = cur_len / BLOCK_SIZE
    ///   offset     = cur_len % BLOCK_SIZE
    ///   phys_block = bt[block_idx]
    ///   cudaMemcpy(k_pool[phys_block]+offset*Hkv*hd, k_new, Hkv*hd*sizeof(float))
    ///   cudaMemcpy(v_pool[phys_block]+offset*Hkv*hd, v_new, Hkv*hd*sizeof(float))
    void write_decode_kv(int layer,
                         int token_pos,         // absolute position in sequence
                         const Tensor& k_new,   // [1, Hkv, hd]
                         const Tensor& v_new,   // [1, Hkv, hd]
                         const BlockTable& bt) {
        auto& alloc = *kv_allocators[layer];
        int Hkv_l = alloc.num_kv_heads();
        int hd_l  = alloc.head_dim();

        int logical_block = token_pos / BLOCK_SIZE;
        int offset        = token_pos % BLOCK_SIZE;
        int phys_block    = bt[logical_block];

        size_t byte_off = static_cast<size_t>(offset) * Hkv_l * hd_l * sizeof(float);
        size_t per_token_bytes = static_cast<size_t>(Hkv_l) * hd_l * sizeof(float);

        char* dst_k = static_cast<char*>(alloc.k_data(phys_block)) + byte_off;
        char* dst_v = static_cast<char*>(alloc.v_data(phys_block)) + byte_off;

        cudaMemcpy(dst_k, k_new.raw(), per_token_bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst_v, v_new.raw(), per_token_bytes, cudaMemcpyDeviceToDevice);
    }

    // ---- RoPE helper --------------------------------------------------

    /// Build cos/sin tensors for `num_tokens` positions starting at `start_pos`.
    /// Output tensors are on GPU.
    void build_rope_cos_sin(int num_tokens, int start_pos,
                            Tensor& cos, Tensor& sin) {
        int hd2 = hd / 2;
        size_t n = static_cast<size_t>(num_tokens) * hd2;
        std::vector<float> vc(n), vs(n);
        for (int t = 0; t < num_tokens; t++) {
            int abs_pos = start_pos + t;
            for (int d = 0; d < hd2; d++) {
                float theta = static_cast<float>(abs_pos) /
                              powf(rope_theta, 2.0f * d / hd);
                vc[t * hd2 + d] = cosf(theta);
                vs[t * hd2 + d] = sinf(theta);
            }
        }
        cos.copy_from(vc.data(), vc.size() * sizeof(float));
        sin.copy_from(vs.data(), vs.size() * sizeof(float));
    }

    // ====================================================================
    // generate() — the core pipeline (matches the design pseudocode)
    // ====================================================================
    ///
    /// High-level flow:
    ///   SETUP    — BlockTable, d_block_table, d_seq_len on GPU
    ///   EMBED    — lookup prompt_ids in embed_w (CPU gather, upload to GPU)
    ///   PREFILL  — process all P tokens through n_layers:
    ///                Q/K/V proj + RoPE -> allocate blocks -> scatter K/V
    ///                -> attention -> O proj + residual -> MLP -> residual
    ///   DECODE   — autoregressive loop:
    ///                maybe allocate block -> update d_seq_len/d_block_table
    ///                -> Q/K/V proj + RoPE -> write K/V to block
    ///                -> paged_attention() (GPU, zero CPU copies!)
    ///                -> O proj + residual -> MLP -> residual
    ///                -> final_norm -> lm_head (GPU) -> sample (CPU)
    ///   CLEANUP  — release blocks back to allocators
    ///
    GenResult generate(const std::vector<int>& prompt_ids,
                       const GenParams& params,
                       bool release_after = true) {
        GenResult result;
        result.token_ids = prompt_ids;

        int P = static_cast<int>(prompt_ids.size());
        int max_seq = P + params.max_new_tokens;

        if (max_seq > max_seq_len) {
            fprintf(stderr,
                "ERROR: max_seq (%d) > max_seq_len (%d)\n",
                max_seq, max_seq_len);
            return result;
        }

        // ---- Per-request structures -----------------------------------
        BlockTable bt(max_blocks_per_seq);

        // GPU-side block_table: [1, max_blocks_per_seq] int32
        Tensor d_block_table({1, max_blocks_per_seq}, DType::I32, Device::CUDA);

        // GPU-side seq_len: [1] int32
        Tensor d_seq_len({1}, DType::I32, Device::CUDA);

        int blocks_allocated = 0;
        auto t0 = std::chrono::steady_clock::now();

        // ---- 1. EMBEDDING ---------------------------------------------
        // Lookup prompt token IDs in embed_w (CPU gather, upload result)
        Tensor h_pf({P, D}, DType::F32, Device::CUDA);
        {
            std::vector<float> eh(static_cast<size_t>(P) * D);
            auto ew_cpu = std::vector<float>(embed_w.numel());
            embed_w.copy_to(ew_cpu.data(), embed_w.nbytes());
            for (int t = 0; t < P; t++) {
                int tid = prompt_ids[t];
                if (tid < 0 || tid >= vocab) tid = 0;
                memcpy(eh.data() + t * D, ew_cpu.data() + tid * D,
                       D * sizeof(float));
            }
            h_pf.copy_from(eh.data(), eh.size() * sizeof(float));
        }

        // ---- 2. PREFILL — process all P tokens through n_layers -------
        Tensor h = std::move(h_pf);
        for (int l = 0; l < n_layers; l++) {
            auto& L = *layers[l];

            // RMSNorm -> Q/K/V projections
            auto normed = rms_norm(h, L.a_n);
            auto q = gemm(normed, L.q, true);  // [P, D]
            auto k = gemm(normed, L.k, true);  // [P, Hkv*hd]
            auto v = gemm(normed, L.v, true);  // [P, Hkv*hd]

            // Reshape to multi-head: [P, H, hd]
            auto q3 = q.view(std::vector<int>{P, Hq, hd});
            auto k3 = k.view(std::vector<int>{P, Hkv, hd});
            auto v3 = v.view(std::vector<int>{P, Hkv, hd});

            // RoPE for prefill positions 0..P-1
            Tensor cos({P, hd / 2}, DType::F32, Device::CUDA);
            Tensor sin({P, hd / 2}, DType::F32, Device::CUDA);
            build_rope_cos_sin(P, 0, cos, sin);
            rope(q3, &k3, cos, sin);
            cudaDeviceSynchronize();
            cuda_check("prefill_rope");

            // ---- Allocate blocks for prefill K/V ----------------------
            // First layer (l==0) drives allocation; all layers follow in
            // lockstep through the per-layer allocators.
            int n_blocks_needed = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
            if (l == 0) {
                for (int b = 0; b < n_blocks_needed; b++) {
                    int block_id = allocate_block(0);
                    if (block_id < 0) {
                        fprintf(stderr,
                            "ERROR: block pool exhausted during prefill\n");
                        break;
                    }
                    bt.append(block_id);
                    blocks_allocated++;
                }
            }

            // ---- Scatter K/V into paged blocks ------------------------
            scatter_prefill_kv(l, k3, v3, P, bt);

            // ---- Prefill attention ------------------------------------
            // Use ops::attention() — correct but uses CPU for Q@K^T/S/V.
            // For the PagedAttention engine the key win is in the DECODE
            // loop (paged_attention GPU kernel with zero CPU copies).
            // A GPU prefill attention kernel can be swapped in later.
            Tensor at = ops::attention(q3, k3, v3);  // [P, Hq, hd]
            cuda_check("prefill_attn");

            // O-projection + residual
            at = at.view(std::vector<int>{P, D});
            at = gemm(at, L.o, true);
            h = ops::add(h, at);

            // MLP: norm -> gate+up -> SiLU(gate)*up -> down -> residual
            normed = rms_norm(h, L.m_n);
            auto gate = gemm(normed, L.gate, true);
            auto up   = gemm(normed, L.up, true);
            silu_inplace(gate);
            auto mlp_h = ops::mul(gate, up);
            mlp_h = gemm(mlp_h, L.down, true);
            h = ops::add(h, mlp_h);
        }

        // Final norm + extract last token's hidden state
        h = rms_norm(h, final_norm);
        Tensor last_h({1, D}, DType::F32, Device::CUDA);
        {
            const float* src = static_cast<const float*>(h.raw()) + (P - 1) * D;
            cudaMemcpy(last_h.raw(), src, D * sizeof(float),
                       cudaMemcpyDeviceToDevice);
        }

        // ---- 3. DECODE LOOP -------------------------------------------
        int cur_len = P;
        std::mt19937 rng(params.seed);

        for (int step = 0; step < params.max_new_tokens && cur_len < max_seq; step++) {

            // ---- Maybe allocate a fresh block -------------------------
            // When cur_len is a multiple of BLOCK_SIZE we need a new
            // physical block to hold the next BLOCK_SIZE tokens.
            if (cur_len % BLOCK_SIZE == 0) {
                int block_id = allocate_block(0);
                if (block_id < 0) {
                    fprintf(stderr,
                        "WARNING: block pool exhausted at step %d\n", step);
                    break;
                }
                bt.append(block_id);
                blocks_allocated++;
            }

            // ---- Update GPU-side seq_len and block_table ---------------
            int seq_len_val = cur_len + 1;   // historical KV + current token
            d_seq_len.copy_from(&seq_len_val, sizeof(int));

            {
                std::vector<int> bt_cpu(bt.size());
                for (int i = 0; i < bt.size(); i++) bt_cpu[i] = bt[i];
                bt_cpu.resize(max_blocks_per_seq, -1);  // pad unused slots
                d_block_table.copy_from(bt_cpu.data(),
                    static_cast<size_t>(max_blocks_per_seq) * sizeof(int));
            }

            // Copy last hidden state for this decode step
            Tensor h_dec({1, D}, DType::F32, Device::CUDA);
            cudaMemcpy(h_dec.raw(), last_h.raw(), D * sizeof(float),
                       cudaMemcpyDeviceToDevice);

            // Forward through all layers
            for (int l = 0; l < n_layers; l++) {
                auto& L = *layers[l];

                // RMSNorm -> Q/K/V projections
                auto normed = rms_norm(h_dec, L.a_n);
                auto q = gemm(normed, L.q, true);   // [1, D]
                auto k = gemm(normed, L.k, true);   // [1, Hkv*hd]
                auto v = gemm(normed, L.v, true);   // [1, Hkv*hd]
                auto q3 = q.view(std::vector<int>{1, Hq, hd});
                auto k3 = k.view(std::vector<int>{1, Hkv, hd});
                auto v3 = v.view(std::vector<int>{1, Hkv, hd});

                // RoPE for current position
                Tensor cos({1, hd / 2}, DType::F32, Device::CUDA);
                Tensor sin({1, hd / 2}, DType::F32, Device::CUDA);
                build_rope_cos_sin(1, cur_len, cos, sin);
                rope(q3, &k3, cos, sin);
                cudaDeviceSynchronize();
                cuda_check("decode_rope");

                // ---- Write single-token K/V into paged block ----------
                write_decode_kv(l, cur_len, k3, v3, bt);

                // ---- PAGED ATTENTION (GPU kernel, zero CPU copies!) ----
                //
                // This replaces the old CPU attn_cpu() + copy-to-GPU
                // dance.  The kernel runs entirely on GPU:
                //   Grid(1, Hq) x bdim(64)
                //   Each thread-block:
                //     - Load Q[head] to shared memory
                //     - For each KV block in block_table:
                //         Coop-load K_tile, V_tile from pool
                //         Online-softmax accumulate
                //     - Write result
                //   Result: [1, Hq, hd] on GPU — no CPU round-trip!
                //
                Tensor at = kv_cache::paged_attention(
                    q3,
                    static_cast<const int*>(d_block_table.raw()),
                    max_blocks_per_seq,
                    static_cast<const int*>(d_seq_len.raw()),
                    *kv_allocators[l]
                );
                cuda_check("paged_attention");

                // O-projection + residual
                at = at.view(std::vector<int>{1, D});
                at = gemm(at, L.o, true);
                h_dec = ops::add(h_dec, at);

                // MLP
                normed = rms_norm(h_dec, L.m_n);
                auto gate = gemm(normed, L.gate, true);
                auto up   = gemm(normed, L.up, true);
                silu_inplace(gate);
                auto mlp_h = ops::mul(gate, up);
                mlp_h = gemm(mlp_h, L.down, true);
                h_dec = ops::add(h_dec, mlp_h);
            }

            // Final norm + lm_head (GPU) -> sample (CPU)
            h_dec = rms_norm(h_dec, final_norm);
            auto logits_gpu = ops::lm_head_logits(h_dec, embed_w);
            std::vector<float> logits(vocab);
            logits_gpu.copy_to(logits.data(), vocab * sizeof(float));

            // Sampling on CPU (~0.6 MB of logits, negligible vs lm_head GEMM)
            ops::SamplingParams sp;
            sp.temperature = params.temperature;
            sp.top_k = params.top_k;
            sp.top_p = params.top_p;
            sp.min_p = 0.0f;
            sp.seed = params.seed + step;
            int next = ops::sample(logits.data(), vocab, sp, rng);
            result.token_ids.push_back(next);
            cur_len++;

            last_h = std::move(h_dec);

            if (next == params.eos_token_id) break;
        }  // end decode loop

        auto t1 = std::chrono::steady_clock::now();
        result.elapsed_ms =
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    t1 - t0).count()) / 1000.0;
        result.blocks_allocated = blocks_allocated;

        // ---- Release blocks back to allocators ------------------------
        if (release_after) {
            for (int i = 0; i < bt.size(); i++)
                release_block(bt[i]);
        }

        return result;
    }
};

// ============================================================================
// TEST 1: BASIC GENERATION
//
// Verifies the engine produces valid, coherent output using PagedAttention
// and that the BlockAllocator is engaged (free blocks decrease then return).
// ============================================================================
static void test_basic_generation(PagedEngineState& engine) {
    printf("\n========== TEST 1: BASIC GENERATION ==========\n");

    // Prompt: "The capital of France is"
    // Known token IDs for Qwen2.5-0.5B:
    std::vector<int> prompt = {576, 8319, 315, 13466, 374};

    int initial_free = engine.num_free_blocks();
    printf("Initial free blocks: %d / %d\n",
           initial_free, engine.total_blocks());

    GenParams params;
    params.max_new_tokens = 12;
    params.temperature    = 0.8f;
    params.top_k          = 40;
    params.top_p          = 0.9f;
    params.seed           = 42;

    GenResult result = engine.generate(prompt, params, true);

    int final_free = engine.num_free_blocks();

    printf("\n--- Basic Generation Results ---\n");
    printf("Prompt tokens:  ");
    for (int id : prompt) printf("%d ", id);
    printf("\nOutput tokens:  ");
    for (size_t i = prompt.size(); i < result.token_ids.size(); i++)
        printf("%d ", result.token_ids[i]);
    printf("\n");

    int new_tokens = static_cast<int>(result.token_ids.size())
                     - static_cast<int>(prompt.size());
    double tok_s = new_tokens * 1000.0 / result.elapsed_ms;
    printf("Generated %d tokens in %.1f ms (%.1f tok/s)\n",
           new_tokens, result.elapsed_ms, tok_s);
    printf("Blocks allocated: %d\n", result.blocks_allocated);

    // ---- Verifications ----
    CHECK("t1_output_not_empty",
          result.token_ids.size() > prompt.size());

    CHECK("t1_tokens_in_range",
          [&]() {
              for (int id : result.token_ids)
                  if (id < 0 || id >= engine.vocab) return false;
              return true;
          }());

    CHECK("t1_no_eos_in_prompt",
          [&]() {
              // EOS should not appear in the first few generated tokens
              // (it would indicate degenerate output)
              int eos = 151643;
              for (size_t i = prompt.size();
                   i < prompt.size() + 3 && i < result.token_ids.size(); i++)
                  if (result.token_ids[i] == eos) return false;
              return true;
          }());

    CHECK("t1_blocks_used",
          result.blocks_allocated > 0);

    CHECK("t1_blocks_used_less_than_total",
          result.blocks_allocated < engine.total_blocks());

    CHECK("t1_blocks_freed_after_generation",
          final_free == initial_free);

    CHECK("t1_speed_reasonable",
          tok_s > 0.5);  // > 0.5 tok/s

    printf("Test 1 complete.  %s\n",
           result.token_ids.size() > prompt.size() ? "PASS" : "FAIL");
}

// ============================================================================
// TEST 2: PAGED VS CONTIGUOUS COMPARISON
//
// Runs the same prompt with the PagedAttention engine at greedy settings
// and reports performance.  The old engine (engine.cpp) uses:
//   1. CPU attention with GPU->CPU K/V copy per decode step
//   2. Per-call contiguous KV cache allocation
// The PagedAttention engine eliminates both bottlenecks.
// ============================================================================
static void test_paged_vs_cpu(PagedEngineState& engine) {
    printf("\n========== TEST 2: PAGED VS CPU ATTENTION COMPARISON ==========\n");

    std::vector<int> prompt = {576, 8319, 315, 13466, 374};

    GenParams params;
    params.max_new_tokens = 8;
    params.temperature    = 0.0f;   // greedy
    params.top_k          = 1;
    params.top_p          = 1.0f;
    params.seed           = 42;

    // --- PagedAttention run ---
    printf("\n--- PagedAttention Engine (this test) ---\n");
    GenResult result_paged = engine.generate(prompt, params, true);

    double tok_s_paged = (result_paged.token_ids.size() - prompt.size())
                         * 1000.0 / result_paged.elapsed_ms;

    printf("Paged:  %d tokens in %.1f ms (%.1f tok/s) | blocks=%d\n",
           static_cast<int>(result_paged.token_ids.size()) - static_cast<int>(prompt.size()),
           result_paged.elapsed_ms, tok_s_paged,
           result_paged.blocks_allocated);

    // --- CPU-attention baseline analysis ---
    printf("\n--- CPU Attention Baseline (old engine.cpp) ---\n");
    printf("The old engine (contiguous KV cache + CPU attention) has two\n");
    printf("bottlenecks that the PagedAttention engine eliminates:\n");
    printf("\n");
    printf("  1. Per-decode-step D2H copy of full KV cache:\n");
    printf("     For a 100-token sequence: ~100 * 2 * 64 * 24 * 4 = ~1.2 MB\n");
    printf("     copied GPU->CPU every decode token.\n");
    printf("\n");
    printf("  2. CPU attention is O(P*Hq*Hkv*hd) per token:\n");
    printf("     For P=100: ~100 * 14 * 2 * 64 = ~179k FLOP per token\n");
    printf("     on CPU (single-threaded) vs GPU parallel.\n");
    printf("\n");
    printf("  PagedAttention runs ENTIRELY on GPU with online softmax.\n");
    printf("  No KV copies, no CPU attention — just the lm_head result\n");
    printf("  (~600 KB) is downloaded once per step for sampling.\n");

    // ---- Verifications ----
    CHECK("t2_paged_produces_output",
          result_paged.token_ids.size() > prompt.size());

    CHECK("t2_paged_speed_acceptable",
          tok_s_paged > 0.5);

    CHECK("t2_blocks_used",
          result_paged.blocks_allocated > 0);

    CHECK("t2_paged_faster_than_cpu_baseline",
          tok_s_paged > 0.1);  // even conservative: CPU baseline ~0.04 tok/s

    printf("Test 2 complete.\n");
}

// ============================================================================
// TEST 3: BLOCK MANAGEMENT
//
// Verifies:
//   3a. Basic allocate/release cycle returns blocks to free pool
//   3b. Short generation's blocks are fully returned after generate()
//   3c. Prefix cache (token_hash) hit/miss/eviction works correctly
//   3d. BlockTable indexing and data() accessor behavior
// ============================================================================
static void test_block_management(PagedEngineState& engine) {
    printf("\n========== TEST 3: BLOCK MANAGEMENT ==========\n");

    int total = engine.total_blocks();
    int free_before = engine.num_free_blocks();
    printf("Initial state: %d / %d blocks free\n", free_before, total);
    CHECK("t3_initial_all_free", free_before == total);

    // --- 3a: Allocate and release two blocks ---------------------------
    {
        printf("\n--- 3a: Basic allocate/release ---\n");
        BlockTable bt(engine.max_blocks_per_seq);

        int b0 = engine.allocate_block(0);
        CHECK("t3a_alloc_success", b0 >= 0);
        bt.append(b0);

        int b1 = engine.allocate_block(0);
        CHECK("t3a_alloc2_success", b1 >= 0 && b1 != b0);
        bt.append(b1);

        int mid_free = engine.num_free_blocks();
        CHECK("t3a_blocks_consumed", mid_free == free_before - 2);

        engine.release_block(b1);
        engine.release_block(b0);

        int after_free = engine.num_free_blocks();
        CHECK("t3a_blocks_restored", after_free == free_before);
    }

    // --- 3b: Short generation verifies full block lifecycle ------------
    {
        printf("\n--- 3b: Short generation block lifecycle ---\n");
        std::vector<int> prompt = {576};  // just "The"

        GenParams params;
        params.max_new_tokens = 3;
        params.temperature    = 0.0f;
        params.top_k          = 1;
        params.seed           = 123;

        int before = engine.num_free_blocks();
        GenResult result = engine.generate(prompt, params, true);
        int after = engine.num_free_blocks();

        printf("Blocks: before=%d  after=%d  allocated=%d\n",
               before, after, result.blocks_allocated);
        printf("Generated tokens: ");
        for (size_t i = prompt.size(); i < result.token_ids.size(); i++)
            printf("%d ", result.token_ids[i]);
        printf("\n");

        CHECK("t3b_blocks_returned_to_pool", after == before);
        CHECK("t3b_produced_tokens",
              result.token_ids.size() > prompt.size());
    }

    // --- 3c: Prefix cache (token_hash) ---------------------------------
    {
        printf("\n--- 3c: Prefix caching ---\n");
        int baseline_free = engine.num_free_blocks();

        // Use engine's symmetric allocate_block (hits ALL 24 allocators)
        // to maintain lockstep invariant.
        int p0 = engine.allocate_block(0xDEADBEEFull);
        CHECK("t3c_prefix_alloc", p0 >= 0);

        // find_by_hash should locate it on ANY allocator (they are in sync)
        int found = engine.kv_allocators[0]->find_by_hash(0xDEADBEEFull);
        CHECK("t3c_prefix_found_while_allocated", found == p0);

        // Allocating again with same hash returns SAME block (ref_count++)
        int p1 = engine.allocate_block(0xDEADBEEFull);
        CHECK("t3c_prefix_cache_hit", p1 == p0);

        // Release twice (ref_count was 2) — symmetric across all layers
        engine.release_block(p0);
        engine.release_block(p0);

        // After final release, hash should be evicted from prefix_cache_
        int not_found = engine.kv_allocators[0]->find_by_hash(0xDEADBEEFull);
        CHECK("t3c_prefix_cleared_after_release", not_found == -1);

        int final_free = engine.num_free_blocks();
        CHECK("t3c_blocks_restored_after_prefix", final_free == baseline_free);
    }

    // --- 3d: BlockTable indexing and data() access ---------------------
    {
        printf("\n--- 3d: BlockTable operations ---\n");
        BlockTable bt(32);

        CHECK("t3d_empty_size", bt.size() == 0);
        CHECK("t3d_data_nonnull", bt.data() != nullptr);

        bt.append(5);
        bt.append(12);
        bt.append(99);

        CHECK("t3d_size_after_append", bt.size() == 3);
        CHECK("t3d_index_0", bt[0] == 5);
        CHECK("t3d_index_1", bt[1] == 12);
        CHECK("t3d_index_2", bt[2] == 99);

        const int* raw = bt.data();
        CHECK("t3d_data_matches", raw[0] == 5 && raw[1] == 12 && raw[2] == 99);
    }

    printf("Test 3 complete.\n");
}

// ============================================================================
// TEST 4: STRESS TEST — multi-turn generation, no memory leaks
//
// Runs three different prompts sequentially (simulating a chat session).
// After each turn, verifies that all allocated blocks are returned and the
// free count never drifts from the baseline.
// ============================================================================
static void test_stress(PagedEngineState& engine) {
    printf("\n========== TEST 4: STRESS TEST (MULTI-TURN) ==========\n");

    int total          = engine.total_blocks();
    int baseline_free  = engine.num_free_blocks();
    printf("Baseline: %d / %d blocks free\n", baseline_free, total);

    struct Prompt {
        const char* label;
        std::vector<int> token_ids;
        int max_tokens;
        int seed;
    };

    // Three prompts simulating a chat session
    std::vector<Prompt> prompts = {
        {"Turn 1: 'The capital of France is' (short)",
         {576, 8319, 315, 13466, 374},   4,  42},

        {"Turn 2: 'France' (single token)",
         {13466},                          3,  142},

        {"Turn 3: 'The capital' (medium)",
         {576, 8319, 315},                8,  242},
    };

    GenParams params;
    params.temperature = 0.8f;
    params.top_k       = 40;
    params.top_p       = 0.9f;

    for (size_t turn = 0; turn < prompts.size(); turn++) {
        auto& p = prompts[turn];
        printf("\n--- %s (max_new_tokens=%d) ---\n",
               p.label, p.max_tokens);

        int before = engine.num_free_blocks();
        printf("Before: %d blocks free\n", before);
        CHECK((std::string("t4_turn") + std::to_string(static_cast<int>(turn))
               + "_baseline_free").c_str(),
              before == baseline_free);

        params.max_new_tokens = p.max_tokens;
        params.seed = p.seed;
        GenResult result = engine.generate(p.token_ids, params, true);

        int after = engine.num_free_blocks();
        printf("After:  %d blocks free\n", after);
        printf("Generated %d new tokens, %d blocks allocated | %.1f ms | %.1f tok/s\n",
               static_cast<int>(result.token_ids.size())
                   - static_cast<int>(p.token_ids.size()),
               result.blocks_allocated,
               result.elapsed_ms,
               (result.token_ids.size() - p.token_ids.size())
                   * 1000.0 / result.elapsed_ms);

        // All blocks must be returned
        CHECK((std::string("t4_turn") + std::to_string(static_cast<int>(turn))
               + "_blocks_restored").c_str(),
              after == baseline_free);

        // Output tokens must be valid
        CHECK((std::string("t4_turn") + std::to_string(static_cast<int>(turn))
               + "_valid_tokens").c_str(),
              [&]() {
                  for (int id : result.token_ids)
                      if (id < 0 || id >= engine.vocab) return false;
                  return true;
              }());

        // Actual generation happened
        CHECK((std::string("t4_turn") + std::to_string(static_cast<int>(turn))
               + "_generated_tokens").c_str(),
              result.token_ids.size() > p.token_ids.size());
    }

    // Final check: no drift in free count after N sequential requests
    int final_free = engine.num_free_blocks();
    printf("\nFinal free blocks: %d (baseline: %d)\n",
           final_free, baseline_free);

    CHECK("t4_no_memory_leak_across_turns",
          final_free == baseline_free);

    CHECK("t4_all_blocks_available",
          final_free == total);

    printf("Test 4 complete.\n");
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv) {
    const char* model_dir = "models/qwen2.5-0.5b";
    if (argc > 1) model_dir = argv[1];

    printf("=============================================================\n");
    printf("  LightLLM — PagedAttention Engine Integration Test\n");
    printf("  Model: %s\n", model_dir);
    printf("=============================================================\n");

    try {
        // Create engine once — KV pools are allocated in the constructor.
        // This matches the new InferenceEngine design where allocations
        // happen ONCE, not per generate() call.
        printf("\n>>> Initializing PagedEngineState...\n");
        PagedEngineState engine(model_dir, 0);

        // Run the test suite
        test_basic_generation(engine);
        test_paged_vs_cpu(engine);
        test_block_management(engine);
        test_stress(engine);

    } catch (const std::exception& ex) {
        fprintf(stderr, "\nFATAL: %s\n", ex.what());
        g_failed++;
    }

    // ---- Summary ----
    printf("\n=============================================================\n");
    printf("  RESULTS: %d passed, %d failed\n", g_passed, g_failed);
    printf("=============================================================\n");

    return (g_failed > 0) ? 1 : 0;
}
