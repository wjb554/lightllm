#pragma once
/// Speculative Decoding: draft model generates k candidate tokens,
/// target model verifies them in one parallel forward pass.
///
/// Algorithm (Leviathan et al., 2023):
///   1. Draft model generates k tokens greedily (very fast)
///   2. Target model does ONE forward pass with [prompt + k draft tokens]
///   3. For each position p, compare target logits with draft token:
///      - If draft token matches target's argmax -> accept
///      - If mismatch -> reject from position p, sample from target
///   4. KV cache: draft tokens are written to target's KV cache
///      Accepted tokens stay, rejected tokens are overwritten
///   5. Repeat: draft from the last accepted position
///
/// Two modes are supported:
///
///   SELF-SPECULATION (draft_model_dir empty):
///     The target model's first `draft_layers` serve as the draft model.
///     When draft_layers == n_layers_ the draft and target are identical —
///     this is a correctness-validation configuration with 100% acceptance
///     (every draft token is guaranteed to be accepted by the target).
///     For actual wall-clock speedup set draft_layers < n_layers_
///     (e.g. n_layers_ / 4).
///
///   DUAL-MODEL (draft_model_dir set):
///     A separate (smaller/faster) draft model generates candidate tokens
///     which the target model verifies in one batched forward pass.

#include <vector>
#include "lightllm/tensor.h"

namespace lightllm {
namespace engine {

struct SpeculativeResult {
    std::vector<int> accepted_tokens;  // tokens accepted by target
    int num_accepted = 0;
    int num_drafted = 0;               // total draft tokens produced
    double draft_time_ms = 0;           // time spent in draft generation
    double verify_time_ms = 0;          // time spent in target verification
};

/// Run one speculative decode step: draft k tokens, verify with target.
///
/// Self-speculation mode: the `draft` parameter is unused.  The draft
/// tokens are produced by forwarding the first 2 layers of `target`.
///
/// When a dedicated draft model is available in the future, pass it
/// as `draft` and the implementation will be upgraded to use it.
///
/// @param target           The target (full) model engine server.
/// @param draft            Reserved for future dual-model support (ignored).
/// @param state            Per-request state (prompt tokens, seq_len,
///                         next_hidden_state, block_tables, etc.).
/// @param num_draft_tokens Number of draft tokens k to generate.
/// @return                 SpeculativeResult with accepted tokens and timing.
SpeculativeResult speculative_step(
    class EngineServer& target,
    class EngineServer* draft,
    class RequestState& state,
    int num_draft_tokens);

}  // namespace engine
}  // namespace lightllm
