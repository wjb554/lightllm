#pragma once
/// Speculative decoding draft-engine abstraction.
///
/// The draft stage is separated from the target engine so it can NEVER write
/// the target's authoritative KV cache.  Two strategies implement the same
/// interface:
///
///   - SelfSpecDraftEngine: the target's first M layers run as a cheap draft,
///     sharing the target's weight tensors but owning its OWN KV cache.
///   - DualModelDraftEngine: a separate smaller draft model (its own engine).
///
/// Position convention (same as EngineServer):
///   After step() appended the base token, state.seq_len == L and the base
///   token occupies position L (its KV is not yet written).  Drafting token
///   at position p predicts position p+1.  draft() produces D1..Dk =
///   predictions at positions L+1..L+k by processing inputs
///   [base, D1, ..., D_{k-1}] at positions L..L+k-1 and writing the draft's
///   own K/V for those positions.

#include <memory>
#include <vector>

#include "lightllm/kv_cache/block_allocator.h"

namespace lightllm {
namespace engine {

class EngineServer;
struct RequestState;

class DraftEngine {
public:
    virtual ~DraftEngine() = default;

    /// Number of transformer layers the draft runs.
    virtual int num_layers() const = 0;
    /// Block size of the draft KV cache (== target BLOCK_SIZE).
    virtual int block_size() const = 0;

    /// Bring the draft KV cache up to `state.seq_len` positions
    /// (positions 0..seq_len-1).  Idempotent — no-op if already synced.
    virtual void sync(RequestState& state) = 0;

    /// Draft k tokens D1..Dk (predictions at seq_len+1..seq_len+k), writing
    /// the draft's own K/V for the draft inputs.  Returns the tokens actually
    /// drafted (fewer than k if EOS was hit).
    ///
    /// @param sample  true → sample from the draft distribution p_d (required
    ///                for Leviathan probability acceptance); false → greedy
    ///                argmax (deterministic, for strict-argmax acceptance).
    /// @param probs   when sample=true and non-null, receives the FULL p_d
    ///                distributions concatenated row-major ([k][vocab]) so the
    ///                orchestrator can compute the residual p_t - p_d.
    virtual std::vector<int> draft(RequestState& state, int k, bool sample,
                                   std::vector<float>* probs = nullptr) = 0;

    /// After the orchestrator accepted N draft tokens, drop draft cache beyond
    /// the last accepted position (sets state.draft_seq_len = state.seq_len).
    /// Stale suffix K/V is never read and is overwritten on the next sync.
    virtual void rollback(RequestState& state) = 0;

    /// True for self-speculation (layers of the target itself).
    virtual bool is_self() const { return false; }

    /// Draft-layer allocators indexed 0..num_layers()-1 (absolute layer index
    /// for self-spec; draft-engine layer index for dual-model).  Used to
    /// release a request's draft blocks.
    virtual std::vector<kv_cache::BlockAllocator*>& allocators() = 0;

    /// Reset the draft KV cache (between benchmark runs).
    virtual void clear_kv_cache() = 0;
};

// ============================================================================
// SelfSpecDraftEngine — target's first M layers, own KV cache
// ============================================================================
class SelfSpecDraftEngine : public DraftEngine {
public:
    /// @param target        the target EngineServer (weights shared, read-only)
    /// @param draft_layers  number of early layers to use as the draft (M)
    SelfSpecDraftEngine(EngineServer& target, int draft_layers);
    ~SelfSpecDraftEngine() override = default;

    int num_layers() const override { return M_; }
    int block_size() const override { return bs_; }
    void sync(RequestState& state) override;
    std::vector<int> draft(RequestState& state, int k, bool sample,
                           std::vector<float>* probs = nullptr) override;
    void rollback(RequestState& state) override;
    bool is_self() const override { return true; }
    std::vector<kv_cache::BlockAllocator*>& allocators() override { return alloc_ptrs_; }
    void clear_kv_cache() override;

private:
    /// Allocate draft blocks (all M layers, lockstep) until `num_tokens`
    /// positions are covered.  Appends to state.draft_block_tables.
    bool ensure_blocks(RequestState& state, int num_tokens);

    /// Embed token ids [T] -> [T, D] via the target's embedding weight.
    Tensor embed(const std::vector<int>& tokens);

    EngineServer& target_;
    int M_;                  // draft layer count
    int bs_;                 // block size
    int num_blocks_per_layer_;
    float eps_;

    // Own KV cache: one allocator per draft layer.
    std::vector<std::unique_ptr<kv_cache::BlockAllocator>> allocs_;
    std::vector<kv_cache::BlockAllocator*> alloc_ptrs_;
};

// ============================================================================
// DualModelDraftEngine — a separate smaller draft model (its own EngineServer)
// ============================================================================
class DualModelDraftEngine : public DraftEngine {
public:
    /// Takes ownership of a fully-constructed draft EngineServer.
    DualModelDraftEngine(std::unique_ptr<EngineServer> draft);
    ~DualModelDraftEngine() override;  // defined in draft_engine.cpp

    int num_layers() const override;
    int block_size() const override;
    void sync(RequestState& state) override;
    std::vector<int> draft(RequestState& state, int k, bool sample,
                           std::vector<float>* probs = nullptr) override;
    void rollback(RequestState& state) override;
    std::vector<kv_cache::BlockAllocator*>& allocators() override { return draft_allocs_; }
    void clear_kv_cache() override;

private:
    bool ensure_blocks(RequestState& state, int num_tokens);
    Tensor embed_single(int token);
    Tensor embed_batch(const std::vector<int>& tokens);

    std::unique_ptr<EngineServer> draft_;
    std::vector<kv_cache::BlockAllocator*> draft_allocs_;
};

}  // namespace engine
}  // namespace lightllm
