/// Attention Divergence Test — paged_attention kernel correctness
///
/// Compares two attention computations on the SAME Q, K, V:
///   PATH A (reference): Materialize K/V from paged blocks, compute GQA
///        attention manually on CPU (independent reference implementation)
///   PATH B (kernel):    Write K/V to cache, call paged_attention GPU kernel
///
/// Both paths use identical Q (single extra token) and identical K/V
/// (prompt tokens from cache + current token's projection).  The only
/// variable is the attention computation itself — any divergence beyond
/// float epsilon signals a kernel bug.
///
/// TEST FLOW:
///   1. Load model (FP32, max_seq_len=128)
///   2. Prefill prompt via step() → prompt K/V in paged cache
///   3. Predict first decode token greedily
///   4. For each layer:
///      a. RMSNorm + Q/K/V projection + RoPE
///      b. Copy Q to CPU; read prior K/V from cache to CPU
///      c. PATH A: manual GQA attention on CPU
///      d. PATH B: write K/V to cache → paged_attention GPU → copy to CPU
///      e. Compare max |PATH_A[d] - PATH_B[d]|
///      f. O-projection + MLP → hidden for next layer
///   5. Token prediction comparison
///   6. Report per-layer results

#include "lightllm/engine/engine.h"
#include "lightllm/engine/scheduler.h"
#include "lightllm/kv_cache/prefix_cache.h"
#include "lightllm/kv_cache/paged_attention.h"
#include "lightllm/kv_cache/block_allocator.h"
#include "lightllm/tokenizer/tokenizer.h"
#include "lightllm/ops/norm.h"
#include "lightllm/ops/lm_head.h"
#include "lightllm/ops/gemm.h"
#include "lightllm/ops/rope.h"
#include "lightllm/ops/activation.h"
#include "lightllm/ops/elementwise.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <limits>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace lightllm;
using namespace lightllm::engine;
using namespace lightllm::kv_cache;
using namespace lightllm::ops;

// ============================================================================
static void enable_utf8_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

// ============================================================================
static int greedy_argmax(const float* logits, int n) {
    int best = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; i++)
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    return best;
}

static float softmax_inplace(float* x, int n) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++) x[i] *= inv;
    }
    return sum;
}

// ============================================================================
// Embed tokens on CPU → F32 GPU tensor [T, D]
// ============================================================================
static Tensor embed_tokens_cpu(const Tensor& embed_w,
                                const std::vector<int>& tokens,
                                int vocab_size, int D) {
    int T = static_cast<int>(tokens.size());
    if (T == 0) return Tensor({0, D}, DType::F32, Device::CUDA);
    size_t row_bytes = static_cast<size_t>(D) * sizeof(float);
    std::vector<float> ew(embed_w.numel());
    embed_w.copy_to(ew.data(), embed_w.nbytes());
    std::vector<float> emb(static_cast<size_t>(T) * D);
    for (int t = 0; t < T; t++) {
        int tid = tokens[t];
        if (tid < 0 || tid >= vocab_size) tid = 0;
        memcpy(emb.data() + static_cast<size_t>(t) * D,
               ew.data() + static_cast<size_t>(tid) * D, row_bytes);
    }
    Tensor h({T, D}, DType::F32, Device::CUDA);
    h.copy_from(emb.data(), emb.size() * sizeof(float));
    return h;
}

// ============================================================================
// Read prior K/V from paged cache to CPU
//   k_out/v_out: [num_prior * Hkv * hd], row-major per [pos, head, dim]
// ============================================================================
static void read_kv_cache_to_cpu(int num_prior,
                                  int Hkv, int hd, int block_size,
                                  const BlockTable& bt,
                                  BlockAllocator& alloc,
                                  std::vector<float>& k_out,
                                  std::vector<float>& v_out) {
    size_t row_bytes = static_cast<size_t>(Hkv) * hd * sizeof(float);
    size_t total = static_cast<size_t>(num_prior) * Hkv * hd;
    k_out.resize(total);
    v_out.resize(total);
    for (int p = 0; p < num_prior; p++) {
        int blk_idx = p / block_size;
        int offset  = p % block_size;
        int phys    = bt[blk_idx];
        const void* sk = static_cast<const char*>(alloc.k_data(phys))
                         + offset * row_bytes;
        const void* sv = static_cast<const char*>(alloc.v_data(phys))
                         + offset * row_bytes;
        cudaMemcpy(k_out.data() + static_cast<size_t>(p) * Hkv * hd,
                   sk, row_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(v_out.data() + static_cast<size_t>(p) * Hkv * hd,
                   sv, row_bytes, cudaMemcpyDeviceToHost);
    }
}

// ============================================================================
// Write single-token K/V [1, Hkv, hd] F32 into paged block at token_pos
// ============================================================================
static void write_kv_to_cache(int token_pos,
                               const Tensor& k_new,
                               const Tensor& v_new,
                               const BlockTable& bt,
                               BlockAllocator& alloc,
                               int block_size) {
    int blk_idx = token_pos / block_size;
    int offset  = token_pos % block_size;
    int phys    = bt[blk_idx];
    int Hkv     = k_new.size(1);
    int hd_val  = k_new.size(2);
    size_t row_bytes = static_cast<size_t>(Hkv) * hd_val * sizeof(float);
    void* dst_k = static_cast<char*>(alloc.k_data(phys)) + offset * row_bytes;
    void* dst_v = static_cast<char*>(alloc.v_data(phys)) + offset * row_bytes;
    cudaMemcpy(dst_k, k_new.raw(), row_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst_v, v_new.raw(), row_bytes, cudaMemcpyDeviceToDevice);
}

// ============================================================================
// Manual GQA attention on CPU (reference)
//   Q_cpu:   [Hq * hd]  row-major per head
//   K_cache: [P * Hkv * hd]  row-major per [pos, head, dim]
//   V_cache: [P * Hkv * hd]
//   Returns: [Hq * hd]
// ============================================================================
static std::vector<float> cpu_attention_ref(
        const std::vector<float>& Q_cpu,
        const std::vector<float>& K_cache,
        const std::vector<float>& V_cache,
        int P, int Hq, int Hkv, int hd) {
    int groups = Hq / Hkv;
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> out(Hq * hd, 0.0f);

    for (int h = 0; h < Hq; h++) {
        int kv = h / groups;
        const float* qh = Q_cpu.data() + h * hd;
        const float* k_base = K_cache.data() + kv * hd;
        const float* v_base = V_cache.data() + kv * hd;

        // scores[p] = dot(Q[h], K[p, kv]) * scale
        std::vector<float> scores(P);
        for (int p = 0; p < P; p++) {
            const float* kp = k_base + p * Hkv * hd;
            float dot = 0.0f;
            for (int d = 0; d < hd; d++) dot += qh[d] * kp[d];
            scores[p] = dot * scale;
        }

        softmax_inplace(scores.data(), P);

        float* out_h = out.data() + h * hd;
        for (int p = 0; p < P; p++) {
            const float* vp = v_base + p * Hkv * hd;
            float w = scores[p];
            for (int d = 0; d < hd; d++) out_h[d] += w * vp[d];
        }
    }
    return out;
}

// ============================================================================
// Build RoPE cos/sin for a single position
// ============================================================================
static void build_rope_cos_sin(int token_pos, int half_hd, float rope_theta,
                                std::vector<float>& vc, std::vector<float>& vs) {
    vc.resize(half_hd);
    vs.resize(half_hd);
    for (int d = 0; d < half_hd; d++) {
        float theta = static_cast<float>(token_pos)
                      / std::pow(rope_theta, 2.0f * d / (half_hd * 2));
        vc[d] = std::cos(theta);
        vs[d] = std::sin(theta);
    }
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    enable_utf8_console();

    const char* model_dir = "models/qwen2.5-0.5b";
    if (argc > 1) model_dir = argv[1];

    printf("=================================================================\n");
    printf("  LightLLM — Attention Divergence Test (kernel correctness)\n");
    printf("  Model: %s\n", model_dir);
    printf("=================================================================\n\n");

    // ---- Tokenizer ----
    char tok_path[512];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", model_dir);
    tokenizer::Tokenizer tok(tok_path);
    printf("[1] Tokenizer: vocab=%d\n", tok.vocab_size());

    // ---- EngineServer: FP32 ----
    EngineServer engine(model_dir, 128, 256, 0,
                        prefix_cache_policy_from_env(), false);
    int vocab    = engine.vocab_size();
    int D        = engine.hidden_dim();
    int n_layers = engine.num_layers();
    int Hkv      = engine.num_kv_heads();
    int hd       = engine.head_dim();
    int Hq       = D / hd;
    int groups   = Hq / Hkv;
    int half_hd  = hd / 2;
    int block_sz = engine.block_size();

    printf("[2] Engine: %d layers, D=%d, Hq=%d, Hkv=%d, hd=%d, vocab=%d, FP32\n",
           n_layers, D, Hq, Hkv, hd, vocab);

    // ---- Prompt ----
    std::string prompt_str = "The capital of France is";
    std::vector<int> prompt_tokens = tok.encode(prompt_str);
    int P = static_cast<int>(prompt_tokens.size());
    printf("[3] Prompt: \"%s\" → %d tokens: [", prompt_str.c_str(), P);
    for (int i = 0; i < P; i++) printf("%s%d", i ? "," : "", prompt_tokens[i]);
    printf("]\n");
    if (P == 0) { fprintf(stderr, "ERROR: empty prompt\n"); return 1; }

    // ---- Scheduler + state ----
    auto scheduler = Scheduler::create(SchedulerPolicy::FCFS, 16);
    Request req;
    req.id = 1; req.prompt_tokens = prompt_tokens;
    req.max_new_tokens = 20; req.eos_token_id = 151643;
    auto state_uptr = engine.create_request_state(req, D);
    std::unordered_map<int, std::unique_ptr<RequestState>> states;
    states[1] = std::move(state_uptr);
    scheduler->add_request(req);

    // ========================================================================
    // PHASE A: Prefill
    // ========================================================================
    printf("\n--- PHASE A: Prefill ---\n");
    {
        auto s = scheduler->step();
        if (s.empty()) { fprintf(stderr, "ERROR: empty prefill step\n"); return 1; }
        engine.step(s, states);
    }
    auto& state = *states[1];
    printf("  Prefill done: seq_len=%d\n", state.seq_len);

    // ========================================================================
    // PHASE B: Predict first decode token
    // ========================================================================
    printf("\n--- PHASE B: Predict first decode token ---\n");
    Tensor emb_prompt = embed_tokens_cpu(engine.embed_weight(),
                                          prompt_tokens, vocab, D);
    Tensor h_prompt_out = engine.forward_n_layers(n_layers, emb_prompt, 0);

    // Hidden state → final_norm → lm_head → greedy
    int T_next = 0;
    {
        Tensor h_last({1, D}, DType::F32, Device::CUDA);
        size_t rb = static_cast<size_t>(D) * sizeof(float);
        cudaMemcpy(h_last.raw(), h_prompt_out.data<float>() + (P - 1) * D,
                   rb, cudaMemcpyDeviceToDevice);
        Tensor hn = rms_norm(h_last, engine.final_norm_weight(),
                             engine.rms_norm_eps());
        Tensor lg = lm_head_logits(hn, engine.embed_weight());
        std::vector<float> lc(vocab);
        lg.copy_to(lc.data(), static_cast<size_t>(vocab) * sizeof(float));
        T_next = greedy_argmax(lc.data(), vocab);
    }
    printf("  Predicted token: %d\n", T_next);

    // Embed extra token → h_cur (input hidden state for layer 0)
    std::vector<int> extra_vec = {T_next};
    Tensor h_extra_emb = embed_tokens_cpu(engine.embed_weight(),
                                           extra_vec, vocab, D);
    Tensor h_cur({1, D}, DType::F32, Device::CUDA);
    cudaMemcpy(h_cur.raw(), h_extra_emb.raw(),
               static_cast<size_t>(D) * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Full all-tokens embedding for PATH A token prediction later
    std::vector<int> all_tokens = prompt_tokens;
    all_tokens.push_back(T_next);
    int T_all = static_cast<int>(all_tokens.size());
    int tgt_pos = T_all - 1;  // = P

    printf("\n--- PHASE C: Setup ---\n");
    printf("  Total tokens: %d, extra position: %d\n", T_all, tgt_pos);

    // Ensure block table covers tgt_pos
    {
        int have = state.block_tables.empty() ? 0 : state.block_tables[0].size();
        int need = tgt_pos / block_sz + 1;
        while (have < need) {
            if (!engine.allocate_decode_blocks(state)) {
                printf("  WARNING: KV cache full\n"); break;
            }
            have++;
        }
        printf("  Block table: %d blocks (need %d)\n", have, need);
    }

    // GPU-side metadata for paged_attention
    int max_blocks = (engine.max_seq_len() + block_sz - 1) / block_sz;
    std::vector<int> bt_init(max_blocks, -1);
    Tensor d_block_table({max_blocks}, DType::I32, Device::CUDA);
    Tensor d_seq_len({1}, DType::I32, Device::CUDA);

    // ========================================================================
    // PHASE D: Per-layer attention comparison
    // ========================================================================
    printf("\n=================================================================\n");
    printf("  PER-LAYER ATTENTION COMPARISON\n");
    printf("  PATH A = CPU reference (manual GQA attention)\n");
    printf("  PATH B = paged_attention GPU kernel\n");
    printf("  Same Q, same K/V for both paths.\n");
    printf("  Threshold = 1.0e-4\n");
    printf("=================================================================\n");
    printf("  Layer | PATH_A max  | PATH_B max  | max_diff     | status\n");
    printf("  ------|-------------|-------------|--------------|-------\n");

    const double threshold = 1e-4;
    int    first_divergent = -1;
    bool   all_match       = true;
    double worst_diff      = 0.0;
    int    worst_layer     = -1;

    for (int L = 0; L < n_layers; L++) {
        const auto& Lw = engine.layer_weights(L);
        auto& alloc = engine.kv_allocator(L);

        // ---- (a) RMSNorm → Q/K/V projection ----
        Tensor normed = rms_norm(h_cur, Lw.a_n, engine.rms_norm_eps());
        Tensor q_flat = gemm(normed, Lw.q, true);  // [1, Hq*hd] F32
        Tensor k_flat = gemm(normed, Lw.k, true);  // [1, Hkv*hd] F32
        Tensor v_flat = gemm(normed, Lw.v, true);  // [1, Hkv*hd] F32
        q_flat.reshape_inplace({1, Hq, hd});
        k_flat.reshape_inplace({1, Hkv, hd});
        v_flat.reshape_inplace({1, Hkv, hd});

        // ---- (b) RoPE ----
        {
            std::vector<float> vc, vs;
            build_rope_cos_sin(tgt_pos, half_hd, engine.rope_theta(), vc, vs);
            Tensor cos({1, half_hd}, DType::F32, Device::CUDA);
            Tensor sin({1, half_hd}, DType::F32, Device::CUDA);
            cos.copy_from(vc.data(), half_hd * sizeof(float));
            sin.copy_from(vs.data(), half_hd * sizeof(float));
            rope(q_flat, &k_flat, cos, sin);
        }

        // ---- (c) Copy Q to CPU (for PATH A) ----
        std::vector<float> Q_cpu(Hq * hd);
        cudaMemcpy(Q_cpu.data(), q_flat.raw(),
                   static_cast<size_t>(Hq) * hd * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // ---- (d) Read prior K/V from cache (PATH A reference) ----
        std::vector<float> K_cache_cpu, V_cache_cpu;
        // Only read if there are prior tokens
        if (tgt_pos > 0) {
            read_kv_cache_to_cpu(tgt_pos, Hkv, hd, block_sz,
                                 state.block_tables[L], alloc,
                                 K_cache_cpu, V_cache_cpu);
        } else {
            K_cache_cpu.clear();
            V_cache_cpu.clear();
        }

        // ---- (e) PATH A: CPU reference attention ----
        // Attends to prior tokens only (no self-attention)
        std::vector<float> out_a = cpu_attention_ref(
            Q_cpu, K_cache_cpu, V_cache_cpu,
            tgt_pos, Hq, Hkv, hd);

        // ---- (f) PATH B: Write K/V to cache → paged_attention ----
        write_kv_to_cache(tgt_pos, k_flat, v_flat,
                          state.block_tables[L], alloc, block_sz);

        // Build GPU block table and seq_len for paged_attention
        {
            auto& bt = state.block_tables[L];
            for (int b = 0; b < bt.size(); b++)
                bt_init[b] = bt[b];
            // unused entries stay -1
            for (int b = bt.size(); b < max_blocks; b++)
                bt_init[b] = -1;
        }
        d_block_table.copy_from(bt_init.data(),
                                 max_blocks * sizeof(int));
        int sl_val = tgt_pos;  // attend to positions 0..tgt_pos-1, NOT including self
        d_seq_len.copy_from(&sl_val, sizeof(int));

        Tensor attn_out_b = paged_attention(
            q_flat,
            d_block_table.data<int>(),
            max_blocks,
            d_seq_len.data<int>(),
            alloc);

        // Copy PATH B attention output to CPU
        std::vector<float> out_b(Hq * hd);
        cudaMemcpy(out_b.data(), attn_out_b.raw(),
                   static_cast<size_t>(Hq) * hd * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // ---- (g) Compare ----
        float max_abs_a = 0.0f, max_abs_b = 0.0f, max_diff = 0.0f;
        for (int i = 0; i < Hq * hd; i++) {
            float va = std::fabs(out_a[i]);
            float vb = std::fabs(out_b[i]);
            float d  = std::fabs(out_a[i] - out_b[i]);
            if (va > max_abs_a) max_abs_a = va;
            if (vb > max_abs_b) max_abs_b = vb;
            if (d  > max_diff)  max_diff  = d;
        }

        bool ok = (static_cast<double>(max_diff) <= threshold);
        if (!ok && first_divergent < 0) {
            first_divergent = L;
            all_match = false;
        }
        if (static_cast<double>(max_diff) > worst_diff) {
            worst_diff  = static_cast<double>(max_diff);
            worst_layer = L;
        }

        printf("   %2d   | %10.6f  | %10.6f  | %11.8f  | %s\n",
               L, (double)max_abs_a, (double)max_abs_b,
               (double)max_diff, ok ? "OK" : "DIVERGE");

        // ---- (h) O-projection + residual + MLP → next hidden state ----
        // Use PATH B's attention output to continue the layer
        {
            Tensor attn_2d = attn_out_b.view({1, D});
            attn_2d = gemm(attn_2d, Lw.o, true);
            h_cur = ops::add(h_cur, attn_2d);

            // MLP
            Tensor m_normed = rms_norm(h_cur, Lw.m_n, engine.rms_norm_eps());
            Tensor gate = gemm(m_normed, Lw.gate, true);
            Tensor up   = gemm(m_normed, Lw.up, true);
            silu_inplace(gate);
            Tensor mlp = ops::mul(gate, up);
            mlp = gemm(mlp, Lw.down, true);
            h_cur = ops::add(h_cur, mlp);
        }
    }

    // ========================================================================
    // PHASE E: Token prediction comparison
    // ========================================================================
    printf("\n=================================================================\n");
    printf("  TOKEN PREDICTION COMPARISON\n");
    printf("=================================================================\n");

    // PATH A (CPU ref) not used for prediction — difference accumulates
    // Compare: PATH A full forward (forward_n_layers on all tokens)
    //          vs  PATH B manual decode (what we just computed)
    {
        // PATH A token prediction
        Tensor h_all_emb = embed_tokens_cpu(engine.embed_weight(),
                                             all_tokens, vocab, D);
        Tensor h_all_out = engine.forward_n_layers(n_layers, h_all_emb, 0);
        Tensor ha({1, D}, DType::F32, Device::CUDA);
        {
            size_t rb = static_cast<size_t>(D) * sizeof(float);
            cudaMemcpy(ha.raw(),
                       h_all_out.data<float>() + tgt_pos * D,
                       rb, cudaMemcpyDeviceToDevice);
        }
        Tensor hn_a = rms_norm(ha, engine.final_norm_weight(),
                                engine.rms_norm_eps());
        Tensor lg_a = lm_head_logits(hn_a, engine.embed_weight());
        std::vector<float> lca(vocab);
        lg_a.copy_to(lca.data(), static_cast<size_t>(vocab) * sizeof(float));
        int pred_a = greedy_argmax(lca.data(), vocab);

        // PATH B token prediction (from our manual forward pass)
        Tensor hb({1, D}, DType::F32, Device::CUDA);
        cudaMemcpy(hb.raw(), h_cur.raw(),
                   static_cast<size_t>(D) * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        Tensor hn_b = rms_norm(hb, engine.final_norm_weight(),
                                engine.rms_norm_eps());
        Tensor lg_b = lm_head_logits(hn_b, engine.embed_weight());
        std::vector<float> lcb(vocab);
        lg_b.copy_to(lcb.data(), static_cast<size_t>(vocab) * sizeof(float));
        int pred_b = greedy_argmax(lcb.data(), vocab);

        printf("  Path A (full causal attn) prediction: token=%d\n", pred_a);
        printf("  Path B (manual decode + paged KV) prediction: token=%d\n", pred_b);
        if (pred_a == pred_b)
            printf("  Token predictions MATCH.\n");
        else
            printf("  Token predictions DIFFER: PathA=%d, PathB=%d\n",
                   pred_a, pred_b);
    }

    // ========================================================================
    // SUMMARY
    // ========================================================================
    printf("\n=================================================================\n");
    if (all_match) {
        printf("  ALL LAYERS MATCH — paged_attention kernel matches CPU reference\n");
        printf("  (max diff across all layers: %.2e at layer %d)\n",
               worst_diff, worst_layer);
    } else {
        printf("  DIVERGENCE at layer %d (threshold %.0e)\n",
               first_divergent, threshold);
        printf("  Worst layer: %d (max diff = %.2e)\n",
               worst_layer, worst_diff);
    }
    printf("=================================================================\n");

    return all_match ? 0 : 1;
}
