#pragma once
/// BatchMainLoop — orchestrates Scheduler + EngineServer for continuous batching.
///
/// LIFECYCLE:
///   1. Create an EngineServer, then construct BatchMainLoop with it.
///   2. Call loop.submit(prompt, max_new, eos_id) for each arriving request.
///   3. Call loop.step() repeatedly (or loop.run() for blocking).
///   4. Each step() calls scheduler.step() -> engine.step() -> processes results.
///   5. Read loop.all_metrics() for per-request latency stats.
///   6. Read loop.step_metrics() for per-step batch efficiency stats.
///
/// METRICS:
///   - BatchMetrics: per-request TTFT, TPOT, total latency, token counts.
///   - BatchStepMetrics: per-step tokens, requests, prefill/decode breakdown.
///
/// BatchMainLoop does NOT own the EngineServer — the caller must ensure it
/// outlives the loop.

#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lightllm/engine/engine.h"
#include "lightllm/engine/scheduler.h"

namespace lightllm {
namespace engine {

// ============================================================================
// BatchMetrics — per-request timing and token metrics
// ============================================================================
struct BatchMetrics {
    int request_id = 0;
    int prompt_len = 0;
    int output_len = 0;

    double arrival_time_sec      = 0.0;   // wall clock when submit() was called
    double first_token_time_sec  = 0.0;   // wall clock when first decode token emitted
    double finish_time_sec       = 0.0;   // wall clock when request finished

    bool finished              = false;
    bool first_token_recorded  = false;

    double ttft_ms() const {
        if (!first_token_recorded) return 0.0;
        return (first_token_time_sec - arrival_time_sec) * 1000.0;
    }
    double latency_ms() const {
        return (finish_time_sec - arrival_time_sec) * 1000.0;
    }
    double tpot_ms() const {
        if (output_len <= 0) return 0.0;
        return (finish_time_sec - first_token_time_sec) * 1000.0
               / static_cast<double>(output_len);
    }
};

// ============================================================================
// BatchStepMetrics — per-step batch efficiency metrics
// ============================================================================
struct BatchStepMetrics {
    int step_number = 0;
    double step_time_sec = 0.0;    // wall clock at start of step
    int tokens_processed = 0;      // total tokens in batch
    int prefill_tokens = 0;        // prefill tokens
    int decode_tokens = 0;         // decode tokens
    int num_requests = 0;          // entries in batch
    int num_prefill_entries = 0;
    int num_decode_entries = 0;
    int active_after = 0;          // active requests remaining after step
};

// ============================================================================
// BatchMainLoop
// ============================================================================
class BatchMainLoop {
public:
    /// Construct with an existing EngineServer and a scheduler policy.
    /// @param engine           already-constructed EngineServer (must outlive loop)
    /// @param policy           scheduling policy (DecodeFirst recommended)
    /// @param chunk_size       max prompt tokens per prefill chunk
    /// @param max_batch_tokens   ceiling on total tokens per batch step
    /// @param max_prefill_entries max prefill entries allowed per batch
    BatchMainLoop(EngineServer& engine,
                  SchedulerPolicy policy = SchedulerPolicy::DecodeFirst,
                  int chunk_size = 16,
                  int max_batch_tokens = 256,
                  int max_prefill_entries = 999);

    /// Destructor — releases all allocated KV cache blocks.
    ~BatchMainLoop();

    /// Submit a new request. Returns the assigned request ID immediately.
    /// @param json_schema  JSON Schema string (empty = unconstrained generation)
    /// @param regex        Regex pattern (empty = unconstrained; json_schema wins if both set)
    int submit(const std::vector<int>& prompt_tokens,
               int max_new_tokens = 64,
               int eos_token_id = 151643,
               const std::string& json_schema = "",
               const std::string& regex = "");

    /// Execute ONE scheduler.step() + engine.step() cycle.
    /// @return true if work was done, false if idle.
    bool step();

    /// Blocking loop: repeatedly call step() until all requests finish.
    void run();

    /// True if any requests are still active (waiting or decoding).
    bool has_active() const;

    /// Access generated tokens for a finished request (empty if still active).
    const std::vector<int>& generated_tokens(int request_id) const;

    /// Number of active + waiting requests.
    int active_count() const;

    // ---- Metrics accessors ----

    const BatchMetrics& metrics(int request_id) const;
    const std::vector<BatchMetrics>& all_metrics() const { return metrics_; }
    const std::vector<BatchStepMetrics>& step_metrics() const { return step_metrics_; }

    /// Aggregate stats over finished requests.
    double avg_ttft_ms() const;
    double avg_tpot_ms() const;
    double avg_latency_ms() const;
    double p50_ttft_ms() const;
    double p50_tpot_ms() const;
    double p50_latency_ms() const;
    double p90_ttft_ms() const;
    double p90_tpot_ms() const;
    double p90_latency_ms() const;
    double p95_ttft_ms() const;
    double p95_tpot_ms() const;
    double p95_latency_ms() const;
    double p99_ttft_ms() const;
    double p99_tpot_ms() const;
    double p99_latency_ms() const;
    double throughput_tok_s() const;

    // ---- Access underlying components ----

    EngineServer& engine() { return engine_; }
    Scheduler& scheduler() { return *scheduler_; }
    int chunk_size() const { return chunk_size_; }
    int total_steps() const { return total_steps_; }
    int total_tokens_generated() const;

    /// Wall clock seconds since construction.
    double wall_clock() const;

private:
    /// Helper: compute percentile from a vector of doubles.
    static double percentile(std::vector<double> values, double p);

    EngineServer& engine_;
    std::unique_ptr<Scheduler> scheduler_;
    int chunk_size_;
    int max_batch_tokens_;

    /// Per-request engine state (owned by the loop).
    std::unordered_map<int, std::unique_ptr<RequestState>> states_;

    /// Per-request metrics.
    std::vector<BatchMetrics> metrics_;
    std::unordered_map<int, size_t> metrics_index_;

    /// Per-step batch efficiency.
    std::vector<BatchStepMetrics> step_metrics_;

    int next_id_ = 0;
    int total_steps_ = 0;

    std::chrono::steady_clock::time_point epoch_;
};

}  // namespace engine
}  // namespace lightllm
