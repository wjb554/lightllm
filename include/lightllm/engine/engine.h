#pragma once
/// LightLLM Inference Engine — heterogeneous batch execution with PagedAttention.
///
/// INTEGRATION POINT with Scheduler:
///   The Scheduler produces a ScheduleStep (heterogeneous batch of prefill chunks
///   and decode tokens).  The Engine consumes it via step(), which runs the full
///   transformer forward pass for ALL tokens in the batch, then returns sampled
///   tokens for decode entries.  Prefill entries update internal KV cache state
///   but produce no output token from that call.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lightllm/tensor.h"
#include "lightllm/model/model_config.h"
#include "lightllm/kv_cache/block_allocator.h"

// Forward declare xgrammar types at GLOBAL scope (the real xgrammar lives in
// ::xgrammar, NOT inside lightllm::engine).
namespace xgrammar { class GrammarMatcher; class GrammarCompiler; struct TokenizerInfo; }

// Forward declare scheduler types (avoid circular include; scheduler.h includes
// and uses Request / ScheduleStep / ScheduleEntry)
namespace lightllm {
namespace engine {

struct Request;
struct ScheduleStep;
struct ScheduleEntry;

}  // namespace engine
}  // namespace lightllm

namespace lightllm {
namespace engine {

// ============================================================================
// SampledToken — result of one decode step for one request
// ============================================================================
struct SampledToken {
    int request_id;   // which request produced this token
    int token_id;     // the sampled token
    bool is_eos;      // true if this is EOS or max_tokens reached (request done)
};

// ============================================================================
// RequestState — engine-maintained persistent state across steps
// ============================================================================
// Owned by the main serve loop, passed into engine.step() by reference.
// The engine reads/writes this struct but does NOT own it — the caller is
// responsible for creating and destroying RequestState instances.
//
// Fields mutated by engine.step():
//   - num_prefilled  (+= chunk_size for prefill entries)
//   - seq_len        (+= num_tokens for all entries)
//   - generated_tokens  (appended for decode entries)
//   - block_tables   (blocks allocated during prefill / decode)
//   - next_hidden_state  (written after every step, read on next decode step)
//   - finished       (set true on EOS or max_tokens)
struct RequestState {
    // Destructor defined in engine_server.cpp (needs xgrammar::GrammarMatcher
    // complete type for unique_ptr deleter).
    ~RequestState();

    int id;                               // matches Request.id from scheduler
    std::vector<int> prompt_tokens;       // original prompt (immutable after creation)
    std::vector<int> generated_tokens;    // tokens generated so far (appended each decode step)

    int seq_len;          // current total length = prompt_tokens.size() + generated_tokens.size()
    int num_prefilled;    // how many prompt tokens have completed prefill (0 initially)
    int num_cached_tokens = 0;  // tokens matched by prefix cache lookup (0 = no match)

    // Per-layer logical-to-physical block mapping.
    // Indexed by layer [0..n_layers-1].  All layers have identical size because
    // we allocate / release in lockstep across every layer's BlockAllocator.
    // The BlockTable stores physical block IDs for the sequence in order.
    // Initialised empty by create_request_state(); blocks are appended during
    // prefill and decode.
    std::vector<kv_cache::BlockTable> block_tables;

    // Hidden state after final_norm from the LAST processed token.
    // Shape [1, D], dtype F32, device CUDA.
    // During prefill: set to the last token's hidden state (input to first decode).
    // During decode:   overwritten by the new token's hidden state.
    Tensor next_hidden_state;

    bool finished = false;   // engine sets true when EOS emitted or max_new_tokens reached
    int max_new_tokens;
    int eos_token_id;

    /// Grammar matcher for constrained decoding (nullptr = unconstrained).
    /// Created during create_request_state() if the request has a schema.
    std::unique_ptr<xgrammar::GrammarMatcher> grammar_matcher;

    /// Buffer for FillNextTokenBitmask output (reused every decode step).
    std::vector<int32_t> grammar_mask;
};

// ============================================================================
// GenerateParams / GenerateResult  (unchanged from original — single-request API)
// ============================================================================
struct GenerateParams {
    int max_new_tokens = 64;
    int eos_token_id   = 151643;   // Qwen EOS
    float temperature  = 1.0f;
    int top_k          = 0;        // 0 = greedy
    float top_p        = 1.0f;     // 1.0 = disabled
    int seed           = 42;
};

struct GenerateResult {
    std::vector<int> token_ids;
    std::string text;
};

// ============================================================================
// InferenceEngine — single-request generation API (backward compatible)
// ============================================================================
class InferenceEngine {
public:
    /// Load model from safetensors + config.
    /// @param model_dir   path to directory with config.json + model.safetensors
    /// @param max_seq_len override max sequence length (0 = use config value)
    InferenceEngine(const std::string& model_dir, int max_seq_len = 0);

    /// Generate tokens from prompt token IDs (single request, no batching yet).
    GenerateResult generate(const std::vector<int>& prompt_ids,
                            const GenerateParams& params = {});

    /// Tokenizer helpers (unchanged).
    std::vector<int> tokenize(const std::string& text) const;
    std::string detokenize(const std::vector<int>& ids) const;

private:
    // ---- Model geometry ----
    model::ModelConfig cfg_;
    int D_, Hq_, Hkv_, hd_, n_layers_, vocab_;

    // ---- Model weights (FP32 on GPU) ----
    Tensor embed_w_, final_norm_;
    struct LayerW { Tensor q,k,v,o,a_n,gate,up,down,m_n; };
    std::vector<std::unique_ptr<LayerW>> layers_;

    // ---- Paged KV Cache ----
    static constexpr int BLOCK_SIZE = 16;
    int max_seq_len_;
    int num_blocks_;
    std::vector<std::unique_ptr<kv_cache::BlockAllocator>> kv_allocators_;

    // ---- Tokenizer data (unchanged) ----
    std::vector<std::string> tok_vocab_;
    void load_vocab(const std::string& path);

    // ---- Helpers ----
    int allocate_block(uint64_t token_hash = 0);
    void release_block(int block_id);

    void scatter_prefill_kv(int layer,
                            const Tensor& k_contig,  // [P, Hkv, hd]
                            const Tensor& v_contig,  // [P, Hkv, hd]
                            int n_tokens,
                            const kv_cache::BlockTable& bt);

    void write_decode_kv(int layer,
                         int token_pos,         // absolute position in sequence
                         const Tensor& k_new,   // [1, Hkv, hd]
                         const Tensor& v_new,   // [1, Hkv, hd]
                         const kv_cache::BlockTable& bt);
};

// ============================================================================
// EngineServer — batched inference engine with heterogeneous step() API
// ============================================================================
// This is the batch-capable engine that the Scheduler drives.  Unlike
// InferenceEngine (which operates on one request at a time), EngineServer
// processes heterogeneous batches containing prefill chunks AND decode tokens
// interleaved in a single forward pass.
//
// LIFECYCLE:
//   1. main loop calls engine.create_request_state(req) when a new request arrives
//   2. main loop calls scheduler.add_request(req)
//   3. scheduler.step() -> ScheduleStep
//   4. engine.step(schedule_step, states) -> vector<SampledToken>
//   5. main loop feeds tokens back, marks finished requests
//   6. engine.release_request(state) when request is done
//
class EngineServer {
public:
    static constexpr int BLOCK_SIZE = 16;

    /// Load model from safetensors + config and pre-allocate KV cache pools.
    /// @param model_dir       path to directory with config.json + model.safetensors
    /// @param max_seq_len     override max sequence length (0 = use config value)
    /// @param max_batch_tokens ceiling on total tokens in a single step
    /// @param kv_cache_mb     KV cache pool size in MB (0 = derive from max_seq_len)
    EngineServer(const std::string& model_dir,
                 int max_seq_len = 0,
                 int max_batch_tokens = 256,
                 int kv_cache_mb = 0,
                 kv_cache::PrefixCachePolicy prefix_cache_policy
                     = kv_cache::prefix_cache_policy_from_env());

    // Defined in engine_server.cpp (needs xgrammar complete types for unique_ptr)
    ~EngineServer();

    /// Create per-request state for a new request.
    /// Allocates NO KV blocks yet — blocks are allocated lazily during the
    /// first prefill step.
    /// @param req  scheduler Request (prompt tokens, max_new_tokens, eos, id)
    /// @param D    hidden dimension (needed to allocate next_hidden_state tensor)
    std::unique_ptr<RequestState> create_request_state(const Request& req, int D);

    /// Execute ONE step of the heterogeneous batch described by `step`.
    ///
    /// Reads and writes per-request state via `states`.  The map is keyed by
    /// request ID (the same ID that appears in ScheduleEntry::request_idx).
    ///
    /// @return SampledToken for each decode entry.  Prefill entries produce NO
    ///         output tokens from this call — they only update internal state.
    std::vector<SampledToken> step(
        const ScheduleStep& step,
        std::unordered_map<int, std::unique_ptr<RequestState>>& states);

    /// Release ALL KV blocks for a finished request across every layer.
    void release_request(RequestState& state);

    // ---- Accessors ----
    int max_seq_len()  const { return max_seq_len_; }
    int block_size()   const { return BLOCK_SIZE; }
    int num_blocks()   const { return num_blocks_; }
    int num_layers()   const { return n_layers_; }
    int hidden_dim()   const { return D_; }
    int vocab_size()   const { return vocab_; }
    int used_blocks()  const;  // blocks in use (layer 0 as representative)
    int free_blocks()  const;  // blocks free (layer 0)

private:
    model::ModelConfig cfg_;
    int D_, Hq_, Hkv_, hd_, n_layers_, vocab_;

    Tensor embed_w_, final_norm_;
    struct EngineLayerW { Tensor q,k,v,o,a_n,gate,up,down,m_n; };
    std::vector<std::unique_ptr<EngineLayerW>> layers_;

    int max_seq_len_;
    int num_blocks_;
    int max_batch_tokens_;
    int max_blocks_per_seq_;
    int kv_cache_mb_ = 0;
    kv_cache::PrefixCachePolicy prefix_cache_policy_;

    // ---- xgrammar infrastructure ----
    std::unique_ptr<xgrammar::GrammarCompiler> grammar_compiler_;
    std::unique_ptr<xgrammar::TokenizerInfo> tokenizer_info_;

    std::vector<std::unique_ptr<kv_cache::BlockAllocator>> kv_allocators_;

    // GPU-side concatenated block tables for batched paged_attention
    Tensor d_all_block_tables_;
    Tensor d_all_seq_lens_;

    // Pre-allocated device-side metadata for batched first-prefill.
    // Reallocated only when N grows beyond current capacity.
    int prefill_batch_capacity_ = 0;
    Tensor d_prefill_kv_offsets_;     // [N] int32
    Tensor d_prefill_offsets_;        // [N] int32
    Tensor d_prefill_num_tokens_;     // [N] int32
    Tensor d_prefill_start_poss_;     // [N] int32
    Tensor d_prefill_token_cumsum_;   // [N+1] int32
    Tensor d_prefill_bt_flat_;        // [N * max_blocks_per_seq_] int32

    // ---- Internal helpers ----
    void scatter_prefill_kv(int layer,
                            const Tensor& k_contig,
                            const Tensor& v_contig,
                            int n_tokens,
                            int start_pos,
                            const std::vector<kv_cache::BlockTable>& block_tables);

    void write_decode_kv(int layer,
                         int token_pos,
                         const Tensor& k_new,
                         const Tensor& v_new,
                         const std::vector<kv_cache::BlockTable>& block_tables);

    Tensor build_input_tensor(
        const ScheduleStep& step,
        const std::unordered_map<int, std::unique_ptr<RequestState>>& states,
        std::vector<std::pair<int, int>>& entry_map);
};

}  // namespace engine
}  // namespace lightllm
