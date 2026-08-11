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
#include "lightllm/engine/draft_engine.h"

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
// SpeculativeStats — per-request speculative decoding counters/timings
// ============================================================================
struct SpeculativeStats {
    int drafted  = 0;   // draft tokens produced
    int accepted = 0;   // draft tokens accepted by the target
    int rejected = 0;   // draft tokens rejected (mismatch / grammar)
    int steps    = 0;   // speculative steps taken
    double draft_ms  = 0.0;   // time in draft generation
    double verify_ms = 0.0;   // time in target verification
    double total_ms  = 0.0;   // total speculation time
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

    /// Pre-sampled first decode token during prefill post-processing.
    /// Set to -1 when no token is pending.  Consumed by build_input_tensor()
    /// on the first decode step, which embeds the token at the correct
    /// RoPE position.
    int pending_first_token = -1;

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

    // Speculative decoding counters
    int spec_drafted  = 0;  // total draft tokens produced
    int spec_accepted = 0;  // total draft tokens accepted

    // Dual-model speculative decoding: draft engine per-request state.
    // draft_block_tables mirrors block_tables but for the draft engine's
    // KV cache.  draft_seq_len tracks how many tokens have been prefilled
    // into the draft engine (always <= seq_len).
    // Initialised empty by create_request_state() for self-spec mode;
    // populated lazily on the first speculative_step() call when a
    // dedicated draft engine is active.
    std::vector<kv_cache::BlockTable> draft_block_tables;
    int draft_seq_len = 0;

    /// Grammar matcher for constrained decoding (nullptr = unconstrained).
    /// Created during create_request_state() if the request has a schema.
    std::unique_ptr<xgrammar::GrammarMatcher> grammar_matcher;

    /// Buffer for FillNextTokenBitmask output (reused every decode step).
    std::vector<int32_t> grammar_mask;

    /// Per-request sampling temperature (copied from Request at creation).
    /// Grammar constrains the token mask; this independently controls sampling.
    float temperature = 1.0f;

    /// Speculative decoding statistics (drafted/accepted/rejected/timings).
    SpeculativeStats spec_stats;
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
// SpeculativeDecodeConfig
// ============================================================================
struct SpeculativeDecodeConfig {
    bool enabled = false;
    std::string draft_model_dir;  // path to draft model (empty = self-speculation)
    int num_draft_tokens = 4;     // k: how many tokens to draft per step
    /// Self-speculation: number of early target layers used as the draft.
    /// 0 = auto (n_layers_ / 4).  Ignored when draft_model_dir is set.
    int draft_layers = 0;
    /// Verify all k draft tokens in one batched (M=k) forward pass vs k
    /// single-token (M=1) calls.  On this codebase the GEMM/attention kernels
    /// are batch-size sensitive: a batched verify numerically diverges from the
    /// sequential single-token draft, collapsing acceptance to ~0% even for a
    /// same-model draft.  Sequential (false) matches the draft's numerics so
    /// acceptance reflects only the draft approximation.  Default false for
    /// correctness; set true to trade acceptance for a faster batched verify.
    bool verify_batch = false;

    /// Leviathan et al. (2023) probability-based acceptance.  Accept a draft
    /// token with probability min(1, p_target(t) / p_draft(t)) and, on
    /// rejection, sample a correction token from the residual distribution
    /// p_target - p_draft (renormalized).  The emitted distribution then
    /// EXACTLY matches the target model's distribution (output is a sample, NOT
    /// greedy-identical).  With a greedy draft p_draft(drafts[j]) = 1, so the
    /// acceptance probability is simply p_target(drafts[j]).
    ///
    /// When false (default): accept only when the target argmax == draft token
    /// — deterministic, output identical to greedy decode, but acceptance
    /// collapses to ~0% for cross-scale dual models (e.g. 0.5B draft → 1.5B).
    bool probability_acceptance = false;
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
                     = kv_cache::prefix_cache_policy_from_env(),
                 bool use_fp16 = false,
                 bool budget_from_free = false);

    // Defined in engine_server.cpp (needs xgrammar complete types for unique_ptr)
    ~EngineServer();

    /// Enable speculative decoding with a draft model.
    /// Must be called BEFORE any requests are processed.
    void enable_speculative_decode(const SpeculativeDecodeConfig& config);

    /// Check if speculative decoding is active.
    bool is_speculative() const { return spec_cfg_.enabled; }

    /// Enable/disable speculative decoding without touching the draft model.
    void set_spec_enabled(bool on) { spec_cfg_.enabled = on; }

    /// Zero all KV cache blocks (call between benchmark runs).
    void clear_kv_cache();

    /// Update draft token count without reloading the draft model.
    void set_num_draft_tokens(int k) { spec_cfg_.num_draft_tokens = k; }

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

    /// Speculative decoding orchestrator: draft + verify + accept for every
    /// active decode request.  Called by BatchMainLoop AFTER step() so the
    /// core decode path stays speculation-free.  Updates state.generated_tokens,
    /// state.seq_len, state.spec_stats, and may set state.finished.
    void speculate(
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

    /// Forward through the first `n` transformer layers (no KV cache write).
    /// Pure compute — processes the given hidden states in one pass without
    /// any paged-attention or KV-cache interaction. Used for draft/verify
    /// in self-speculative decoding.
    /// @param n         number of layers to run (clamped to [1, num_layers()]).
    /// @param h         input hidden states [T, D], dtype F32, on CUDA.
    /// @param start_pos absolute position of the first token (for RoPE).
    /// @return          hidden states after n layers [T, D], F32 on CUDA.
    Tensor forward_n_layers(int n, const Tensor& h, int start_pos) const;

    /// Process k tokens through ALL layers, writing K/V to cache.
    /// Unlike forward_n_layers(), this updates the persistent KV cache
    /// so that subsequent decode steps see the new tokens.
    /// @param tokens       [k, D] hidden states on GPU (weight_dtype_)
    /// @param block_tables per-layer block table for this request
    /// @param start_pos    absolute position of first token (for RoPE + KV write)
    /// @return             [k, D] hidden states after all layers (pre-final-norm)
    Tensor forward_incremental(const Tensor& tokens,
        const std::vector<kv_cache::BlockTable>& block_tables,
        int start_pos);

    /// Process k tokens of ONE sequence through ALL layers in a single
    /// batched pass, writing authoritative K/V to the target cache.
    ///
    /// This is the verification primitive for speculative decoding: it uses
    /// the SAME batched kernels as step()'s decode path, so the K/V it writes
    /// is numerically consistent with normal decoding (fixes the KV-divergence
    /// that corrupted output identity).
    ///
    /// Row i occupies absolute position `start_pos + i`, attends to positions
    /// 0..start_pos+i-1 (decode convention, self excluded), and its K/V is
    /// written BEFORE the batched attention so later rows see earlier KV.
    ///
    /// @param tokens       [k, D] hidden states on GPU (weight_dtype_)
    /// @param block_tables per-layer block table for this sequence
    /// @param start_pos    absolute position of the first token
    /// @param write_kv     false = pure compute (RoPE + paged attention over
    ///                     existing KV, no cache writes)
    /// @return             [k, D] hidden states after all layers (pre-final-norm)
    Tensor forward_batched(const Tensor& tokens,
        const std::vector<kv_cache::BlockTable>& block_tables,
        int start_pos,
        bool write_kv = true);

    /// Core batched layer runner shared by forward_partial, forward_batched
    /// and the speculative draft engines.  Processes [T, D] hidden states
    /// through layers [start_layer, start_layer+num_layers) with paged
    /// attention; when write_kv, writes K/V to `allocs_abs[l]` (indexed by
    /// absolute layer) using `block_tables_abs[l]`.  A draft engine passes its
    /// OWN allocators so the target's authoritative cache is never touched.
    Tensor run_layers(const Tensor& hidden,
                      int start_layer,
                      int num_layers,
                      int start_pos,
                      const std::vector<kv_cache::BlockTable>& block_tables_abs,
                      std::vector<kv_cache::BlockAllocator*>& allocs_abs,
                      bool write_kv);

    /// Allocate one block across all layers for a decode position.
    /// Appends to state.block_tables. Must be called when crossing a
    /// block boundary during incremental forward passes.
    /// @return true if blocks were allocated, false if KV cache is full.
    bool allocate_decode_blocks(RequestState& state);

    /// Access embedding weight (for lm_head operations).
    const Tensor& embed_weight() const { return embed_w_; }

    /// Access final layer norm weight.
    const Tensor& final_norm_weight() const { return final_norm_; }

    /// RMSNorm epsilon from model config.
    float rms_norm_eps() const { return cfg_.rms_norm_eps; }

    /// Access the weight dtype (F32 or F16) for intermediate activations.
    DType weight_dtype() const { return weight_dtype_; }

    /// Number of key-value heads (for GQA reshape).
    int num_kv_heads() const { return Hkv_; }

    /// Head dimension.
    int head_dim() const { return hd_; }

    /// Run a range of transformer layers on a single-token hidden state.
    /// Uses paged-attention for decode, writes K/V to cache at token_pos.
    /// @param hidden       [1, D] input hidden state (weight_dtype_)
    /// @param start_layer  first layer index (0-based)
    /// @param num_layers   number of layers to run
    /// @param block_tables per-layer block tables for this request
    /// @param token_pos    absolute RoPE position and KV write position
    /// @param attn_seq_len number of prior tokens for attention (0..attn_seq_len-1)
    /// @return             [1, D] hidden state after the last processed layer
    Tensor forward_partial(const Tensor& hidden,
                           int start_layer,
                           int num_layers,
                           const std::vector<kv_cache::BlockTable>& block_tables,
                           int token_pos,
                           int attn_seq_len);

    /// Access the KV cache allocator for a given layer (0-based).
    /// Exposed for testing — callers can read/write K/V via k_data()/v_data().
    kv_cache::BlockAllocator& kv_allocator(int layer) {
        return *kv_allocators_[layer];
    }
    const kv_cache::BlockAllocator& kv_allocator(int layer) const {
        return *kv_allocators_[layer];
    }

    /// Per-layer weight bundle (exposed for testing).
    struct EngineLayerW { Tensor q,k,v,o,a_n,gate,up,down,m_n; };

    /// Access the weights for a given layer (0-based).
    const EngineLayerW& layer_weights(int layer) const {
        return *layers_[layer];
    }

    /// Access RoPE theta from model config (for testing).
    float rope_theta() const { return cfg_.rope_theta; }

private:
    model::ModelConfig cfg_;
    int D_, Hq_, Hkv_, hd_, n_layers_, vocab_;

    Tensor embed_w_, final_norm_;
    std::vector<std::unique_ptr<EngineLayerW>> layers_;

    int max_seq_len_;
    int num_blocks_;
    int max_batch_tokens_;
    int max_blocks_per_seq_;
    int kv_cache_mb_ = 0;
    bool use_fp16_ = false;
    /// When true, the KV pool budget is taken from the ACTUAL free GPU memory
    /// at construction (for a draft model created co-resident with a target),
    /// instead of the device-total-minus-weights heuristic for a standalone
    /// engine.
    bool budget_from_free_ = false;

    /// DType for intermediate activations (hidden states, norm outputs, GEMM
    /// intermediates).  Derived from use_fp16_ in the constructor.
    /// - F32: all activations F32 (default, safe for all paths)
    /// - F16: activations use F16 where supported (norm, GEMM, elementwise);
    ///        attention stays F32 with automatic F16<->F32 conversions.
    DType weight_dtype_ = DType::F32;

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

    // ---- Speculative Decoding ----
    SpeculativeDecodeConfig spec_cfg_;
    /// Active draft strategy (self-spec layer-slice or dual-model engine).
    /// Created by enable_speculative_decode(); owns its own KV cache.
    std::unique_ptr<DraftEngine> draft_engine_;

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
                         const std::vector<kv_cache::BlockTable>& block_tables,
                         kv_cache::BlockAllocator& alloc);

    Tensor build_input_tensor(
        const ScheduleStep& step,
        const std::unordered_map<int, std::unique_ptr<RequestState>>& states,
        std::vector<std::pair<int, int>>& entry_map);
};

}  // namespace engine
}  // namespace lightllm
