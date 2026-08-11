/// Speculative Decoding — dual-model speculation with KV-cache paged attention.
///
/// DRAFT:
///   The draft engine generates k tokens auto-regressively using
///   forward_partial (single-token paged attention + KV cache write).
///   The draft engine's KV cache is synced with the target's sequence
///   before drafting begins.
///
/// VERIFY:
///   The target verifies all k draft tokens in ONE batch via
///   forward_incremental (loops over k tokens with paged attention),
///   writing K/V to the target's persistent cache.
///   Acceptance: compare target greedy argmax vs draft token at each
///   position; stop at first mismatch.
///
/// Both paths use the SAME attention mechanism (paged_attention kernel),
/// so same-model self-speculation naturally produces 100% acceptance.

#include "lightllm/engine/speculative_decoder.h"
#include "lightllm/engine/engine.h"
#include "lightllm/ops/sampling.h"
#include "lightllm/ops/lm_head.h"
#include "lightllm/ops/norm.h"
#include "lightllm/ops/embedding.h"
#include "lightllm/ops/argmax.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace lightllm {
namespace engine {

using namespace ops;

// ============================================================================
// Helpers
// ============================================================================

/// Embed a single token using the GPU embedding kernel.
static Tensor embed_single(const Tensor& embed_w, int token,
                           int vocab_size, int D, DType out_dtype) {
    (void)D; (void)out_dtype;  // output dtype is inferred from embed_w
    int id = (token >= 0 && token < vocab_size) ? token : 0;
    Tensor ids({1}, DType::I32, Device::CUDA);
    ids.copy_from(&id, sizeof(int));
    return ops::embedding(ids, embed_w);
}

/// Embed a batch of tokens using the GPU embedding kernel.
static Tensor embed_batch(const Tensor& embed_w,
                          const std::vector<int>& tokens,
                          int vocab_size, int D, DType out_dtype) {
    (void)D; (void)out_dtype;
    int T = static_cast<int>(tokens.size());
    std::vector<int> clamped(T);
    for (int i = 0; i < T; i++) {
        int id = tokens[i];
        clamped[i] = (id >= 0 && id < vocab_size) ? id : 0;
    }
    Tensor ids({T}, DType::I32, Device::CUDA);
    ids.copy_from(clamped.data(), T * sizeof(int));
    return ops::embedding(ids, embed_w);
}

/// Ensure `block_tables` has enough blocks to cover `num_tokens` positions.
/// Returns true on success, false if KV cache is exhausted.
static bool ensure_blocks(EngineServer& engine,
                          std::vector<kv_cache::BlockTable>& block_tables,
                          int num_tokens) {
    int bs = engine.block_size();
    int blocks_needed = (num_tokens + bs - 1) / bs;
    int n_layers = engine.num_layers();
    int have = block_tables.empty() ? 0 : block_tables[0].size();

    while (have < blocks_needed) {
        int bid = engine.kv_allocator(0).allocate(0);
        if (bid < 0) return false;
        for (int l = 1; l < n_layers; l++) {
            engine.kv_allocator(l).allocate(0);
        }
        for (int l = 0; l < n_layers; l++) {
            block_tables[l].append(bid);
        }
        have++;
    }
    return true;
}

// ============================================================================
// speculative_step — dual-model speculative decode
// ============================================================================

SpeculativeResult speculative_step(
    EngineServer& target,
    EngineServer* draft,
    RequestState& state,
    int num_draft_tokens) {

    SpeculativeResult result;

    // ---- Guards ----
    if (num_draft_tokens <= 0 || !draft) return result;
    if (state.finished) return result;

    int D          = target.hidden_dim();
    int vocab      = target.vocab_size();
    int n_layers   = target.num_layers();
    int seq_len    = state.seq_len;
    int block_size = target.block_size();

    int draft_L     = draft->num_layers();
    int draft_D     = draft->hidden_dim();
    DType draft_dt  = draft->weight_dtype();
    DType target_dt = target.weight_dtype();

    const Tensor& draft_ew    = draft->embed_weight();
    const Tensor& draft_fn    = draft->final_norm_weight();
    float draft_eps           = draft->rms_norm_eps();

    // Clamp k
    int max_len = target.max_seq_len();
    int k = num_draft_tokens;
    if (seq_len + k >= max_len) {
        k = max_len - seq_len - 1;
    }
    if (k <= 0) return result;

    auto t0 = std::chrono::steady_clock::now();

    // =========================================================================
    // PHASE 0 — Sync draft KV cache via forward_incremental
    // =========================================================================
    // The draft engine's KV cache must be populated with the same token
    // sequence as the target.  After the RoPE convention fix, step()'s
    // decode path already writes KV consistent with forward_partial, so
    // only the draft engine needs syncing (no target overwrite needed).
    //
    // On the first call (draft_seq_len == 0): full prefill for draft.
    // On subsequent calls: only sync new tokens (accepted since last step).

    Tensor draft_last_hidden;

    if (state.draft_seq_len < seq_len) {
        int n_new = seq_len - state.draft_seq_len;
        int start = state.draft_seq_len;

        // Collect tokens at positions [start, seq_len-1]
        std::vector<int> new_tokens;
        new_tokens.reserve(n_new);
        int P = static_cast<int>(state.prompt_tokens.size());
        for (int i = start; i < seq_len; i++) {
            if (i < P)
                new_tokens.push_back(state.prompt_tokens[i]);
            else
                new_tokens.push_back(state.generated_tokens[i - P]);
        }

        // ---- Sync DRAFT engine ----
        if (!ensure_blocks(*draft, state.draft_block_tables, seq_len)) {
            fprintf(stderr, "spec_decode: draft KV cache exhausted (need %d tokens)\n", seq_len);
            state.draft_seq_len = seq_len;
            return result;
        }
        Tensor h_prefill_d = embed_batch(draft_ew, new_tokens,
                                          draft->vocab_size(), draft_D, draft_dt);
        Tensor h_out_d = draft->forward_incremental(
            h_prefill_d, state.draft_block_tables, start);
        int last_idx = n_new - 1;
        draft_last_hidden = Tensor({1, draft_D}, draft_dt, Device::CUDA);
        size_t d_row_bytes = static_cast<size_t>(draft_D) * dtype_size(draft_dt);
        cudaMemcpy(draft_last_hidden.raw(),
                   static_cast<const char*>(h_out_d.raw()) + last_idx * d_row_bytes,
                   d_row_bytes, cudaMemcpyDeviceToDevice);

        // Target KV cache is already consistent with forward_partial
        // convention (Fix 1: step() now uses correct RoPE positions).
        // No overwrite needed — just sync draft_seq_len.

        state.draft_seq_len = seq_len;
    } else {
        // Draft is in sync — recompute last hidden state from the last
        // generated token.  This is a single forward_partial call (O(draft_L)).
        int last_token = state.generated_tokens.back();
        int pos = seq_len - 1;

        Tensor h_emb = embed_single(draft_ew, last_token, draft->vocab_size(),
                                     draft_D, draft_dt);
        draft_last_hidden = draft->forward_partial(
            h_emb, 0, draft_L, state.draft_block_tables, pos, pos);
    }

    // =========================================================================
    // PHASE 1 — Draft: generate k tokens auto-regressively
    // =========================================================================
    // Pre-allocate all blocks needed for k draft positions (seq_len..seq_len+k-1).
    // This avoids per-iteration allocation inside the hot loop.
    if (!ensure_blocks(*draft, state.draft_block_tables, seq_len + k)) {
        // Draft KV cache exhausted — return whatever we have (nothing yet).
        state.draft_seq_len = seq_len;
        return result;
    }

    std::vector<int> draft_tokens;
    draft_tokens.reserve(k);

    Tensor cur_hidden = std::move(draft_last_hidden);

    for (int i = 0; i < k; i++) {
        // Predict next token from current hidden state
        Tensor h_norm = rms_norm(cur_hidden, draft_fn, draft_eps);
        Tensor logits_gpu = lm_head_logits(h_norm, draft_ew);
        int token = ops::argmax(logits_gpu);

        draft_tokens.push_back(token);
        if (token == state.eos_token_id) break;

        // Prepare for next iteration: embed token and run through draft
        int pos = seq_len + i;  // position of THIS draft token

        Tensor h_emb = embed_single(draft_ew, token, draft->vocab_size(),
                                     draft_D, draft_dt);
        cur_hidden = draft->forward_partial(
            h_emb, 0, draft_L, state.draft_block_tables, pos, pos);
    }

    auto t1 = std::chrono::steady_clock::now();
    result.draft_time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.num_drafted = static_cast<int>(draft_tokens.size());

    if (draft_tokens.empty()) return result;

    // =========================================================================
    // PHASE 2 — Verify: one batch forward_incremental for ALL K draft tokens
    // =========================================================================
    // Embed [tok[seq_len-1], D0, D1, ..., D_{K-2}] and process at positions
    // [seq_len-1, seq_len, ..., seq_len+K-2].  Output[j] at position
    // seq_len-1+j → predicts D_j → compare with draft_tokens[j].
    //
    // This matches the reference algorithm: the target verifies all K draft
    // tokens in ONE batched forward pass through the same paged_attention
    // path that the draft used.

    int actual_k = static_cast<int>(draft_tokens.size());

    // Build verify input: [tok at seq_len-1, D0, D1, ..., D_{K-2}]
    std::vector<int> verify_input;
    verify_input.reserve(actual_k);
    // First token: the token at position seq_len-1
    {
        int P = static_cast<int>(state.prompt_tokens.size());
        int idx = seq_len - 1;
        if (idx < P)
            verify_input.push_back(state.prompt_tokens[idx]);
        else
            verify_input.push_back(state.generated_tokens[idx - P]);
    }
    // Then D0..D_{K-2}
    for (int i = 0; i < actual_k - 1; i++) {
        verify_input.push_back(draft_tokens[i]);
    }

    Tensor h_verify = embed_batch(target.embed_weight(), verify_input,
                                   vocab, D, target_dt);

    Tensor h_out = target.forward_incremental(
        h_verify, state.block_tables, seq_len - 1);

    // ---- Acceptance check ----
    {
        size_t row_bytes = static_cast<size_t>(D) * dtype_size(target_dt);

        for (int j = 0; j < actual_k; j++) {
            // h_out[j] at position seq_len-1+j → predicts token at seq_len+j
            // This is where draft_tokens[j] should live.
            Tensor h_pos({1, D}, target_dt, Device::CUDA);
            cudaMemcpy(h_pos.raw(),
                       static_cast<const char*>(h_out.raw()) + j * row_bytes,
                       row_bytes, cudaMemcpyDeviceToDevice);

            Tensor h_norm = rms_norm(h_pos,
                                     target.final_norm_weight(),
                                     target.rms_norm_eps());
            Tensor logits_gpu = lm_head_logits(h_norm, target.embed_weight());
            int target_token = ops::argmax(logits_gpu);

            if (target_token == draft_tokens[j]) {
                result.accepted_tokens.push_back(draft_tokens[j]);
            } else {
                break;
            }
        }
    }

    auto t2 = std::chrono::steady_clock::now();
    result.verify_time_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    result.num_accepted = static_cast<int>(result.accepted_tokens.size());

    // Sync draft_seq_len to match target after caller applies accepted tokens
    state.draft_seq_len = seq_len + result.num_accepted;

    return result;
}

}  // namespace engine
}  // namespace lightllm
