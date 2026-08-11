/// SelfSpecDraftEngine — the target's first M layers as a cheap draft, with
/// its OWN KV cache so it never corrupts the target's authoritative cache.
///
/// Position convention (see draft_engine.h): state.seq_len == L after step()
/// appended the base token (base occupies position L, KV not yet written).
/// The draft processes [base, D1, ..., D_{k-1}] at positions L..L+k-1,
/// predicting D1..Dk at L+1..L+k, and writes the draft's own K/V.

#include "lightllm/engine/draft_engine.h"
#include "lightllm/engine/engine.h"
#include "lightllm/ops/embedding.h"
#include "lightllm/ops/norm.h"
#include "lightllm/ops/lm_head.h"
#include "lightllm/ops/argmax.h"
#include "lightllm/ops/sampling.h"

#include <cuda_runtime.h>
#include <random>

namespace lightllm {
namespace engine {

using namespace ops;

SelfSpecDraftEngine::SelfSpecDraftEngine(EngineServer& target, int draft_layers)
    : target_(target),
      M_(draft_layers),
      bs_(target.block_size()),
      num_blocks_per_layer_(target.num_blocks()),
      eps_(target.rms_norm_eps())
{
    allocs_.reserve(M_);
    for (int l = 0; l < M_; l++) {
        allocs_.push_back(std::make_unique<kv_cache::BlockAllocator>(
            num_blocks_per_layer_, bs_, target.num_kv_heads(), target.head_dim(),
            kv_cache::PrefixCachePolicy::Hash, DType::F32));
    }
    alloc_ptrs_.reserve(M_);
    for (int l = 0; l < M_; l++) alloc_ptrs_.push_back(allocs_[l].get());
    printf("SelfSpecDraftEngine: %d draft layers, own KV (%d blocks/layer)\n",
           M_, num_blocks_per_layer_);
}

bool SelfSpecDraftEngine::ensure_blocks(RequestState& state, int num_tokens) {
    int blocks_needed = (num_tokens + bs_ - 1) / bs_;
    int have = state.draft_block_tables.empty()
                   ? 0
                   : static_cast<int>(state.draft_block_tables[0].size());
    while (have < blocks_needed) {
        int bid = allocs_[0]->allocate(0);
        if (bid < 0) return false;
        for (int l = 1; l < M_; l++) allocs_[l]->allocate(0);
        for (int l = 0; l < M_; l++) state.draft_block_tables[l].append(bid);
        have++;
    }
    return true;
}

Tensor SelfSpecDraftEngine::embed(const std::vector<int>& tokens) {
    int T = static_cast<int>(tokens.size());
    int vocab = target_.vocab_size();
    std::vector<int> clamped(T);
    for (int i = 0; i < T; i++)
        clamped[i] = (tokens[i] >= 0 && tokens[i] < vocab) ? tokens[i] : 0;
    Tensor ids({T}, DType::I32, Device::CUDA);
    ids.copy_from(clamped.data(), T * sizeof(int));
    return ops::embedding(ids, target_.embed_weight());
}

void SelfSpecDraftEngine::sync(RequestState& state) {
    int L = state.seq_len;
    if (state.draft_seq_len >= L) return;
    int start = state.draft_seq_len;
    int n_new = L - start;
    if (n_new <= 0) return;

    if (!ensure_blocks(state, L)) {
        fprintf(stderr, "SelfSpecDraftEngine: KV exhausted during sync (need %d)\n", L);
        return;
    }

    // Copy the target's AUTHORITATIVE K/V for positions [start, L) and layers
    // 0..M-1 into the draft's own cache.  The draft IS the target's first M
    // layers, so its history cache must contain exactly what the target sees.
    // Recomputing it through a different kernel path (e.g. run_layers paged
    // attention vs step()'s prefill/decode kernels) makes the draft's hidden
    // states diverge from the target's, so the draft never matches the target
    // argmax (0% acceptance).  Copying guarantees the shared prefix is bitwise
    // identical, so acceptance reflects only the M-layer approximation.
    size_t row_bytes = static_cast<size_t>(target_.num_kv_heads())
                       * target_.head_dim() * sizeof(float);
    for (int p = start; p < L; p++) {
        int blk = p / bs_, off = p % bs_;
        for (int l = 0; l < M_; l++) {
            auto& ta = target_.kv_allocator(l);
            auto& da = *allocs_[l];
            int t_bid = state.block_tables[l][blk];
            int d_bid = state.draft_block_tables[l][blk];
            void* t_k = static_cast<char*>(ta.k_data(t_bid)) + off * row_bytes;
            void* t_v = static_cast<char*>(ta.v_data(t_bid)) + off * row_bytes;
            void* d_k = static_cast<char*>(da.k_data(d_bid)) + off * row_bytes;
            void* d_v = static_cast<char*>(da.v_data(d_bid)) + off * row_bytes;
            cudaMemcpy(d_k, t_k, row_bytes, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_v, t_v, row_bytes, cudaMemcpyDeviceToDevice);
        }
    }
    state.draft_seq_len = L;
}

std::vector<int> SelfSpecDraftEngine::draft(RequestState& state, int k, bool sample,
                                            std::vector<float>* probs) {
    std::vector<int> out;
    if (k <= 0 || state.finished) return out;
    int L = state.seq_len;
    int eos = state.eos_token_id;
    int vocab = target_.vocab_size();
    std::mt19937 rng(42);

    if (!ensure_blocks(state, L + k)) {
        state.draft_seq_len = L;
        return out;
    }

    // Base token occupies position L; drafting it predicts D1 at L+1.
    int input_token = state.generated_tokens.back();
    for (int i = 0; i < k; i++) {
        int pos = L + i;
        Tensor h = embed({input_token});
        Tensor h_out = target_.run_layers(h, 0, M_, pos,
                                          state.draft_block_tables, alloc_ptrs_,
                                          /*write_kv=*/true);
        Tensor hn = rms_norm(h_out, target_.final_norm_weight(), eps_);
        Tensor lg = ops::lm_head_logits(hn, target_.embed_weight());

        int pred;
        if (sample) {
            // Sample from the draft distribution p_d (required for Leviathan
            // probability acceptance) and record the full p_d for the residual.
            std::vector<float> pd(vocab);
            lg.copy_to(pd.data(), vocab * sizeof(float));
            detail::softmax_inplace(pd.data(), vocab);
            pred = detail::multinomial_draw(pd.data(), vocab, rng);
            if (probs) probs->insert(probs->end(), pd.begin(), pd.end());
        } else {
            pred = ops::argmax(lg);
        }
        out.push_back(pred);
        if (pred == eos) break;
        input_token = pred;
    }
    state.draft_seq_len = L + static_cast<int>(out.size());
    return out;
}

void SelfSpecDraftEngine::rollback(RequestState& state) {
    // Drop the draft cache beyond the last accepted position.  Stale suffix
    // K/V is never read (attention is bounded by per-position seq_len) and is
    // overwritten on the next sync.
    state.draft_seq_len = state.seq_len;
}

void SelfSpecDraftEngine::clear_kv_cache() {
    for (auto& a : allocs_) a->reset();
}

// ============================================================================
// DualModelDraftEngine
// ============================================================================

DualModelDraftEngine::DualModelDraftEngine(std::unique_ptr<EngineServer> draft)
    : draft_(std::move(draft)) {
    draft_allocs_.reserve(draft_->num_layers());
    for (int l = 0; l < draft_->num_layers(); l++)
        draft_allocs_.push_back(&draft_->kv_allocator(l));
    printf("DualModelDraftEngine: %d layers\n", draft_->num_layers());
}

DualModelDraftEngine::~DualModelDraftEngine() = default;

int DualModelDraftEngine::num_layers() const { return draft_->num_layers(); }
int DualModelDraftEngine::block_size() const { return draft_->block_size(); }

bool DualModelDraftEngine::ensure_blocks(RequestState& state, int num_tokens) {
    int bs = draft_->block_size();
    int n_layers = draft_->num_layers();
    int blocks_needed = (num_tokens + bs - 1) / bs;
    int have = state.draft_block_tables.empty()
                   ? 0
                   : static_cast<int>(state.draft_block_tables[0].size());
    while (have < blocks_needed) {
        int bid = draft_->kv_allocator(0).allocate(0);
        if (bid < 0) return false;
        for (int l = 1; l < n_layers; l++) draft_->kv_allocator(l).allocate(0);
        for (int l = 0; l < n_layers; l++) state.draft_block_tables[l].append(bid);
        have++;
    }
    return true;
}

Tensor DualModelDraftEngine::embed_single(int token) {
    int vocab = draft_->vocab_size();
    int id = (token >= 0 && token < vocab) ? token : 0;
    Tensor ids({1}, DType::I32, Device::CUDA);
    ids.copy_from(&id, sizeof(int));
    return ops::embedding(ids, draft_->embed_weight());
}

Tensor DualModelDraftEngine::embed_batch(const std::vector<int>& tokens) {
    int T = static_cast<int>(tokens.size());
    int vocab = draft_->vocab_size();
    std::vector<int> clamped(T);
    for (int i = 0; i < T; i++)
        clamped[i] = (tokens[i] >= 0 && tokens[i] < vocab) ? tokens[i] : 0;
    Tensor ids({T}, DType::I32, Device::CUDA);
    ids.copy_from(clamped.data(), T * sizeof(int));
    return ops::embedding(ids, draft_->embed_weight());
}

void DualModelDraftEngine::sync(RequestState& state) {
    int L = state.seq_len;
    if (state.draft_seq_len >= L) return;
    int start = state.draft_seq_len;
    int n_new = L - start;
    if (n_new <= 0) return;

    std::vector<int> tokens;
    tokens.reserve(n_new);
    int P = static_cast<int>(state.prompt_tokens.size());
    for (int i = start; i < L; i++) {
        if (i < P)
            tokens.push_back(state.prompt_tokens[i]);
        else
            tokens.push_back(state.generated_tokens[i - P]);
    }

    if (!ensure_blocks(state, L)) return;
    Tensor h = embed_batch(tokens);
    // One batched pass through the draft's own layers writes its KV.
    draft_->forward_batched(h, state.draft_block_tables, start, /*write_kv=*/true);
    state.draft_seq_len = L;
}

std::vector<int> DualModelDraftEngine::draft(RequestState& state, int k, bool sample,
                                             std::vector<float>* probs) {
    std::vector<int> out;
    if (k <= 0 || state.finished) return out;
    int L = state.seq_len;
    int eos = state.eos_token_id;
    int draft_L = draft_->num_layers();
    int vocab = draft_->vocab_size();
    std::mt19937 rng(42);

    if (!ensure_blocks(state, L + k)) {
        state.draft_seq_len = L;
        return out;
    }

    int input_token = state.generated_tokens.back();
    for (int i = 0; i < k; i++) {
        int pos = L + i;
        Tensor h = embed_single(input_token);
        Tensor h_out = draft_->forward_partial(h, 0, draft_L,
                                               state.draft_block_tables, pos, pos);
        Tensor hn = ops::rms_norm(h_out, draft_->final_norm_weight(),
                                  draft_->rms_norm_eps());
        Tensor lg = ops::lm_head_logits(hn, draft_->embed_weight());

        int pred;
        if (sample) {
            std::vector<float> pd(vocab);
            lg.copy_to(pd.data(), vocab * sizeof(float));
            detail::softmax_inplace(pd.data(), vocab);
            pred = detail::multinomial_draw(pd.data(), vocab, rng);
            if (probs) probs->insert(probs->end(), pd.begin(), pd.end());
        } else {
            pred = ops::argmax(lg);
        }
        out.push_back(pred);
        if (pred == eos) break;
        input_token = pred;
    }
    state.draft_seq_len = L + static_cast<int>(out.size());
    return out;
}

void DualModelDraftEngine::rollback(RequestState& state) {
    state.draft_seq_len = state.seq_len;
}

void DualModelDraftEngine::clear_kv_cache() {
    draft_->clear_kv_cache();
}

}  // namespace engine
}  // namespace lightllm
