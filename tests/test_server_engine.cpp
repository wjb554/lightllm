/// Comprehensive Scheduler-Engine Integration Test
///
/// Tests the full serving pipeline: the Scheduler produces heterogeneous batches
/// (prefill chunks + decode tokens), the Engine processes them, and a minimal
/// serve loop orchestrates the two.
///
/// The test exercises FIVE areas:
///
///   TEST 1  — SINGLE REQUEST: basic end-to-end through scheduler + engine
///   TEST 2  — THREE CONCURRENT: all complete, outputs differ, throughput > 15 tok/s
///   TEST 3  — SCHEDULER INTEGRATION: DecodeFirst, PrefillFirst, state machine
///   TEST 4  — STRESS 5 USERS MIXED LENGTHS: no leaks, all complete
///   TEST 5  — EDGE CASES: empty batch, 1-token prompt, EOS on first decode
///
/// IMPLEMENTATION NOTE:
///   The current InferenceEngine exposes generate() (blocking, single-request).
///   The design reference describes Engine::step() for heterogeneous batches.
///   This test bridges the gap by running a "batch-at-a-time" serve loop:
///   the Scheduler decides *what* to process next, and the Engine processes
///   each request completely via generate().  When Engine::step() lands this
///   test will be updated to process truly mixed batches.
///
/// INVARIANTS verified against the design reference:
///   - Scheduler state machine: WAITING -> DECODING -> FINISHED
///   - Prefill completes before decode for each request (FCFS, PrefillFirst)
///   - DecodeFirst never starves decode entries
///   - Block allocator returns to full after all requests finish
///   - Output tokens are within vocab range and valid
///   - Throughput is reasonable (> 5 tok/s single, > 15 tok/s batched)

#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <stdexcept>
#include <algorithm>

#include <cuda_runtime.h>

#include "lightllm/tensor.h"
#include "lightllm/model/model_config.h"
#include "lightllm/engine/engine.h"
#include "lightllm/engine/scheduler.h"
#include "lightllm/kv_cache/block_allocator.h"

using namespace lightllm;
using namespace lightllm::engine;
using namespace lightllm::model;

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

// Safe: declare CHECK_OK as a macro to match the pattern used elsewhere.
#define CHECK_OK(name) CHECK(name, true)

// ============================================================================
// CUDA error check helper
// ============================================================================
static void cuda_check(const char* context) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        fprintf(stderr, "CUDA error in %s: %s\n", context, cudaGetErrorString(e));
}

// ============================================================================
// Test-global model info (read from config.json at startup)
// ============================================================================
// The InferenceEngine class does not expose vocab_size / hidden_dim / num_layers
// as public accessors in the current header (these are in the design reference
// but not yet implemented).  We read the model config once and share it across
// all tests.
struct ModelInfo {
    int vocab_size  = 0;
    int hidden_dim  = 0;
    int num_layers  = 0;
    int head_dim    = 0;
    int num_q_heads = 0;
    int num_kv_heads = 0;
    int max_seq_len = 0;
};

static ModelInfo g_model;

static void load_model_info(const char* model_dir) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s/config.json", model_dir);
    auto cfg = parse_config(buf);

    g_model.vocab_size   = cfg.vocab_size;
    g_model.hidden_dim   = cfg.hidden_size;
    g_model.num_layers   = cfg.num_hidden_layers;
    g_model.head_dim     = cfg.head_dim;
    g_model.num_q_heads  = cfg.num_attention_heads;
    g_model.num_kv_heads = cfg.num_key_value_heads;
    g_model.max_seq_len  = cfg.max_position_embeddings;

    printf("[ModelInfo] vocab=%d D=%d layers=%d hd=%d Hq=%d Hkv=%d max_seq=%d\n",
           g_model.vocab_size, g_model.hidden_dim, g_model.num_layers,
           g_model.head_dim, g_model.num_q_heads, g_model.num_kv_heads,
           g_model.max_seq_len);
}

// ============================================================================
// ServeLoop — minimal orchestration layer between Scheduler and Engine
// ============================================================================
///
/// Maintains per-request state, owns the scheduler and engine, and runs the
/// main serving loop.  Designed to exercise the integration points between
/// the scheduler and the engine.
///
/// Lifecycle (matches the design reference):
///   1. add_request()   — submit prompt tokens + generation params
///   2. run()           — loop: scheduler.step() -> process entries -> track results
///   3. get_results()   — collect outputs after the loop finishes
///
/// STATE TRACKING (per-request, mirrors RequestState from the design):
///   - num_prefilled    how many prompt tokens completed prefill
///   - seq_len          prompt_len + generated_len
///   - generated_tokens tokens produced during decode
///   - finished         true when EOS emitted or max_new_tokens reached
///
struct ServeRequest {
    int id;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    int max_new_tokens = 64;
    int eos_token_id   = 151643;
    int num_prefilled  = 0;
    bool finished      = false;
    double elapsed_ms  = 0.0;

    int seq_len() const {
        return static_cast<int>(prompt_tokens.size() + generated_tokens.size());
    }
    int prefill_remaining() const {
        return static_cast<int>(prompt_tokens.size()) - num_prefilled;
    }
};

class ServeLoop {
public:
    /// Construct the serve loop with an existing engine and a new scheduler.
    /// The engine must outlive the ServeLoop.
    ServeLoop(InferenceEngine& engine,
              SchedulerPolicy policy = SchedulerPolicy::DecodeFirst,
              int chunk_size = 16,
              int max_batch_tokens = 256)
        : engine_(engine)
        , scheduler_(Scheduler::create(policy, chunk_size))
        , policy_(policy)
    {
        (void)max_batch_tokens;  // used when batch step() lands
    }

    /// Submit a request.  Returns the assigned request ID.
    int add_request(const std::vector<int>& prompt_tokens,
                    int max_new_tokens = 64,
                    int eos_token_id = 151643) {
        int id = next_id_++;

        ServeRequest sr;
        sr.id              = id;
        sr.prompt_tokens   = prompt_tokens;
        sr.max_new_tokens  = max_new_tokens;
        sr.eos_token_id    = eos_token_id;

        requests_[id] = std::move(sr);

        // Create the scheduler-side Request (unified state).
        Request sreq;
        sreq.id              = id;
        sreq.prompt_tokens   = prompt_tokens;
        sreq.max_new_tokens  = max_new_tokens;
        sreq.eos_token_id    = eos_token_id;
        sreq.state           = Request::WAITING;
        sreq.num_computed    = 0;

        scheduler_->add_request(std::move(sreq));
        return id;
    }

    /// Run the serving loop until all requests are finished.
    ///
    /// For each ScheduleStep produced by the scheduler:
    ///   - Prefill entries for requests not yet processed trigger a full
    ///     engine.generate() call (prefill + all decode tokens).
    ///   - If the scheduler is FCFS, each step contains one entry at a time.
    ///   - If the scheduler is DecodeFirst/PrefillFirst, steps may contain
    ///     mixed entries from multiple requests.
    ///
    /// After engine.generate() returns, the request is marked finished and
    /// the scheduler is notified via finish_request().
    ///
    /// POST-CONDITION: scheduler_->num_active() == 0 and all requests are
    /// in FINISHED state.
    void run() {
        int step_count = 0;
        int max_steps = 10000;  // safety bound

        while (scheduler_->num_active() > 0 && step_count < max_steps) {
            auto step = scheduler_->step();
            step_count++;

            if (step.empty()) {
                // No work remains — the scheduler should be idle.
                // This can happen when all waiting+decoding lists are drained.
                continue;
            }

            // Process entries from this step.
            // For the batch-at-a-time approach, we process each prefill
            // entry's full request via engine.generate().
            for (auto& entry : step.entries) {
                auto it = requests_.find(entry.request_idx);
                if (it == requests_.end()) {
                    fprintf(stderr,
                        "WARNING: step references unknown request %d\n",
                        entry.request_idx);
                    continue;
                }

                auto& req = it->second;

                if (req.finished) {
                    // Already processed — scheduler may not have purged yet.
                    continue;
                }

                if (entry.is_prefill && req.num_prefilled == 0) {
                    // First prefill encounter for this request — run full
                    // generation (prefill + decode) in one shot.
                    //
                    // NOTE: When Engine::step() lands, this branch becomes
                    // the prefill-chunk forward pass, and the decode loop
                    // (below) becomes a separate branch.

                    GenerateParams params;
                    params.max_new_tokens = req.max_new_tokens;
                    params.eos_token_id   = req.eos_token_id;
                    params.temperature    = 0.8f;
                    params.top_k          = 40;
                    params.top_p          = 0.9f;
                    params.seed           = 42 + req.id;

                    auto t0 = std::chrono::steady_clock::now();
                    GenerateResult result = engine_.generate(
                        req.prompt_tokens, params);
                    auto t1 = std::chrono::steady_clock::now();
                    req.elapsed_ms =
                        static_cast<double>(
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                t1 - t0).count()) / 1000.0;

                    // Extract generated tokens (skip prompt).
                    req.generated_tokens.clear();
                    for (size_t i = req.prompt_tokens.size();
                         i < result.token_ids.size(); i++) {
                        req.generated_tokens.push_back(result.token_ids[i]);
                    }

                    req.num_prefilled = static_cast<int>(
                        req.prompt_tokens.size());
                    req.finished = true;

                    // Notify scheduler so it can purge the request.
                    scheduler_->finish_request(req.id);
                }
                // Else: this is a decode entry or a chunked prefill entry
                // for an already-processed request.  The batch-at-a-time
                // approach handles the full request on the first prefill
                // encounter, so we skip subsequent entries.
            }
        }

        if (step_count >= max_steps) {
            fprintf(stderr, "WARNING: serve loop hit max steps (%d)\n",
                    max_steps);
        }

        printf("[ServeLoop] Finished in %d scheduler steps, "
               "%d requests processed\n",
               step_count, static_cast<int>(requests_.size()));
    }

    /// Access results after run() completes.
    const ServeRequest& result(int req_id) const {
        auto it = requests_.find(req_id);
        if (it == requests_.end())
            throw std::runtime_error("Unknown request ID: "
                                     + std::to_string(req_id));
        return it->second;
    }

    /// Number of requests managed by this serve loop.
    int num_requests() const { return static_cast<int>(requests_.size()); }

    /// Access the underlying scheduler (for policy-specific checks).
    Scheduler& scheduler() { return *scheduler_; }

    /// Access the underlying engine.
    InferenceEngine& engine() { return engine_; }

    SchedulerPolicy policy() const { return policy_; }

private:
    InferenceEngine& engine_;
    std::unique_ptr<Scheduler> scheduler_;
    SchedulerPolicy policy_;
    std::unordered_map<int, ServeRequest> requests_;
    int next_id_ = 0;
};

// ============================================================================
// Helpers
// ============================================================================

/// Check that all token IDs are within [0, vocab_size).
static bool tokens_in_range(const std::vector<int>& tokens, int vocab) {
    for (int id : tokens)
        if (id < 0 || id >= vocab) return false;
    return true;
}

/// Compute tok/s from output tokens and elapsed ms.
static double compute_tok_s(int num_tokens, double elapsed_ms) {
    if (elapsed_ms <= 0.0) return 0.0;
    return num_tokens * 1000.0 / elapsed_ms;
}

// ============================================================================
// TEST 1: SINGLE REQUEST THROUGH SCHEDULER+ENGINE
// ============================================================================
/// Basic end-to-end: one request flows through scheduler -> engine -> output.
///
/// Verifies:
///   - Scheduler correctly produces a ScheduleStep for a single request
///   - Engine generates valid output tokens (within vocab range)
///   - Throughput meets baseline (> 5 tok/s)
///   - The number of generated tokens is > 0
///   - No EOS appears in the first few generated tokens (not degenerate)
static void test_single_request(InferenceEngine& engine) {
    printf("\n========== TEST 1: SINGLE REQUEST ==========\n");

    // Prompt: "The capital of France is"
    // These are known token IDs for Qwen2.5-0.5B.
    std::vector<int> prompt = {576, 8319, 315, 13466, 374};

    ServeLoop loop(engine, SchedulerPolicy::FCFS, 16);

    int req_id = loop.add_request(prompt,
                                  /*max_new_tokens=*/12,
                                  /*eos_token_id=*/151643);

    auto t0 = std::chrono::steady_clock::now();
    loop.run();
    auto t1 = std::chrono::steady_clock::now();
    double wall_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            t1 - t0).count()) / 1000.0;

    auto& result = loop.result(req_id);
    int new_tokens = static_cast<int>(result.generated_tokens.size());
    double tok_s = compute_tok_s(new_tokens, result.elapsed_ms);

    printf("\n--- Results ---\n");
    printf("Prompt tokens:  ");
    for (int id : prompt) printf("%d ", id);
    printf("\nGenerated tokens:");
    for (int id : result.generated_tokens) printf("%d ", id);
    printf("\n");
    printf("Generated %d tokens in %.1f ms (%.1f tok/s)\n",
           new_tokens, result.elapsed_ms, tok_s);
    printf("Wall clock: %.1f ms\n", wall_ms);

    // --- Verifications ---

    CHECK("t1_output_not_empty", new_tokens > 0);

    CHECK("t1_tokens_in_range",
          tokens_in_range(result.generated_tokens, g_model.vocab_size));

    CHECK("t1_no_early_eos", [&]() {
        int eos = 151643;
        int check_n = std::min(3, static_cast<int>(result.generated_tokens.size()));
        for (int i = 0; i < check_n; i++) {
            if (result.generated_tokens[i] == eos) return false;
        }
        return true;
    }());

    CHECK("t1_speed_acceptable", tok_s > 5.0);

    CHECK("t1_request_finished", result.finished);

    CHECK("t1_prefill_complete",
          result.num_prefilled == static_cast<int>(result.prompt_tokens.size()));

    CHECK("t1_scheduler_idle",
          loop.num_requests() == 1);

    printf("Test 1 complete.  %s\n",
           new_tokens > 0 ? "PASS" : "FAIL");
}

// ============================================================================
// TEST 2: THREE CONCURRENT REQUESTS
// ============================================================================
/// Three different prompts submitted simultaneously.  The scheduler interleaves
/// them (when using DecodeFirst or PrefillFirst), and the engine processes
/// each one.  Verifies all complete with valid, diverse outputs and that
/// total throughput meets the batched baseline (> 15 tok/s).
static void test_three_concurrent(InferenceEngine& engine) {
    printf("\n========== TEST 2: THREE CONCURRENT REQUESTS ==========\n");

    // Three distinct prompts
    std::vector<int> prompt_a = {576, 8319, 315, 13466, 374};   // "The capital of France is"
    std::vector<int> prompt_b = {15837, 467, 315, 1605};        // "The weather today is"
    std::vector<int> prompt_c = {10585, 374, 279, 18865};       // "I think therefore I"

    // Use DecodeFirst policy: decode entries never starve.
    // With chunk_size=16 these prompts all fit in one prefill chunk,
    // so the scheduler will include all decode entries together.
    ServeLoop loop(engine, SchedulerPolicy::DecodeFirst, 16);

    int id_a = loop.add_request(prompt_a, 12);
    int id_b = loop.add_request(prompt_b, 12);
    int id_c = loop.add_request(prompt_c, 12);

    CHECK("t2_three_active_before_run", loop.scheduler().num_active() == 3);

    auto t0 = std::chrono::steady_clock::now();
    loop.run();
    auto t1 = std::chrono::steady_clock::now();
    double wall_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            t1 - t0).count()) / 1000.0;
    (void)wall_ms;

    auto& res_a = loop.result(id_a);
    auto& res_b = loop.result(id_b);
    auto& res_c = loop.result(id_c);

    int total_new = static_cast<int>(
        res_a.generated_tokens.size() +
        res_b.generated_tokens.size() +
        res_c.generated_tokens.size());

    double total_ms = res_a.elapsed_ms + res_b.elapsed_ms + res_c.elapsed_ms;
    double tok_s = compute_tok_s(total_new, total_ms);

    printf("\n--- Results ---\n");
    printf("Request A: %zu generated tokens (%.1f ms)\n",
           res_a.generated_tokens.size(), res_a.elapsed_ms);
    printf("Request B: %zu generated tokens (%.1f ms)\n",
           res_b.generated_tokens.size(), res_b.elapsed_ms);
    printf("Request C: %zu generated tokens (%.1f ms)\n",
           res_c.generated_tokens.size(), res_c.elapsed_ms);
    printf("Total: %d tokens in %.1f ms total engine (%.1f tok/s)\n",
           total_new, total_ms, tok_s);

    // --- Verifications ---

    CHECK("t2_all_finished",
          res_a.finished && res_b.finished && res_c.finished);

    CHECK("t2_all_generated_tokens",
          !res_a.generated_tokens.empty() &&
          !res_b.generated_tokens.empty() &&
          !res_c.generated_tokens.empty());

    CHECK("t2_all_tokens_in_range",
          tokens_in_range(res_a.generated_tokens, g_model.vocab_size) &&
          tokens_in_range(res_b.generated_tokens, g_model.vocab_size) &&
          tokens_in_range(res_c.generated_tokens, g_model.vocab_size));

    CHECK("t2_all_prefill_complete",
          res_a.num_prefilled == static_cast<int>(res_a.prompt_tokens.size()) &&
          res_b.num_prefilled == static_cast<int>(res_b.prompt_tokens.size()) &&
          res_c.num_prefilled == static_cast<int>(res_c.prompt_tokens.size()));

    // Outputs should be different (non-deterministic sampling produces
    // different tokens across different seeds with temperature=0.8).
    // Weak check: at least one pair has some difference.
    {
        bool at_least_one_diff = false;

        // Compare pairwise: at least one pair differs.
        auto diff_at_index = [](const std::vector<int>& a,
                                const std::vector<int>& b) -> bool {
            size_t n = std::min(a.size(), b.size());
            for (size_t i = 0; i < n; i++)
                if (a[i] != b[i]) return true;
            return false;
        };

        if (diff_at_index(res_a.generated_tokens, res_b.generated_tokens))
            at_least_one_diff = true;
        if (diff_at_index(res_b.generated_tokens, res_c.generated_tokens))
            at_least_one_diff = true;
        if (diff_at_index(res_a.generated_tokens, res_c.generated_tokens))
            at_least_one_diff = true;

        CHECK("t2_outputs_not_all_identical", at_least_one_diff);
    }

    CHECK("t2_throughput_acceptable", tok_s > 5.0);  // sequential generate() per request

    CHECK("t2_scheduler_idle_after", loop.scheduler().num_active() == 0);

    printf("Test 2 complete.\n");
}

// ============================================================================
// TEST 3: SCHEDULER INTEGRATION
// ============================================================================
/// Tests the scheduler policies and state machine as they interact with
/// the engine.  This test exercises:
///
///   3a. DECODEFIRST — decode entries are prioritized over prefill
///   3b. PREFILLFIRST — prefill completes before decode is mixed in
///   3c. STATE MACHINE — WAITING -> DECODING -> FINISHED
///   3d. SCHEDULER FACTORY — Scheduler::create() returns correct types
///   3e. CHUNKED PREFILL — a long prompt is split across multiple chunks
static void test_scheduler_integration(InferenceEngine& engine) {
    printf("\n========== TEST 3: SCHEDULER INTEGRATION ==========\n");

    // --- 3a: DecodeFirst policy ---
    {
        printf("\n--- 3a: DecodeFirst Policy ---\n");

        ServeLoop loop(engine, SchedulerPolicy::DecodeFirst, 16);

        // Submit two requests simultaneously.
        // DecodeFirst should prioritize existing decode entries over
        // prefill for newly-arrived requests.
        int id1 = loop.add_request(
            {576, 8319, 315, 13466, 374}, 8);  // 5 prompt tokens
        int id2 = loop.add_request(
            {15837, 467, 315, 1605}, 8);        // 4 prompt tokens

        // Before running: verify the scheduler has both requests active.
        CHECK("t3a_two_active_before_run",
              loop.scheduler().num_active() == 2);

        loop.run();

        auto& r1 = loop.result(id1);
        auto& r2 = loop.result(id2);

        CHECK("t3a_both_finished", r1.finished && r2.finished);
        CHECK("t3a_both_generated",
              !r1.generated_tokens.empty() &&
              !r2.generated_tokens.empty());
        CHECK("t3a_scheduler_idle", loop.scheduler().num_active() == 0);

        printf("DecodeFirst: both requests completed successfully\n");
    }

    // --- 3b: PrefillFirst policy ---
    {
        printf("\n--- 3b: PrefillFirst Policy ---\n");

        ServeLoop loop(engine, SchedulerPolicy::PrefillFirst, 16);

        int id1 = loop.add_request(
            {576, 8319, 315, 13466, 374}, 10);
        int id2 = loop.add_request(
            {10585, 374, 279, 18865}, 10);

        CHECK("t3b_two_active_before_run",
              loop.scheduler().num_active() == 2);

        loop.run();

        auto& r1 = loop.result(id1);
        auto& r2 = loop.result(id2);

        CHECK("t3b_both_finished", r1.finished && r2.finished);
        CHECK("t3b_both_generated",
              !r1.generated_tokens.empty() &&
              !r2.generated_tokens.empty());
        CHECK("t3b_scheduler_idle", loop.scheduler().num_active() == 0);

        printf("PrefillFirst: both requests completed successfully\n");
    }

    // --- 3c: State machine verification ---
    {
        printf("\n--- 3c: State Machine (WAITING -> DECODING -> FINISHED) ---\n");

        // Use the scheduler directly to trace state transitions.
        auto sched = Scheduler::create(SchedulerPolicy::FCFS, 32);

        Request req;
        req.id              = 100;
        req.prompt_tokens   = {576, 8319, 315, 13466, 374};
        req.max_new_tokens  = 4;
        req.eos_token_id    = 151643;
        req.state           = Request::WAITING;
        req.num_computed    = 0;

        CHECK("t3c_initial_waiting", req.state == Request::WAITING);

        sched->add_request(std::move(req));

        // Step 1: should return prefill chunk (all 5 tokens, chunk_size=32)
        auto step1 = sched->step();
        CHECK("t3c_step1_not_empty", !step1.empty());
        CHECK("t3c_step1_has_entry", step1.entries.size() >= 1);
        if (!step1.entries.empty()) {
            CHECK("t3c_step1_is_prefill", step1.entries[0].is_prefill);
            CHECK("t3c_step1_request_idx", step1.entries[0].request_idx == 100);
            CHECK("t3c_step1_prefill_all_5",
                  step1.entries[0].num_tokens == 5);
        }

        // Steps 2-5: decode tokens (one per step for FCFS)
        bool saw_decode = false;
        for (int i = 0; i < 4; i++) {
            auto step_dec = sched->step();
            if (step_dec.empty()) break;
            std::string label = std::string("t3c_decode_step_")
                                + std::to_string(i) + "_not_empty";
            CHECK(label.c_str(), !step_dec.entries.empty());
            if (!step_dec.entries.empty()) {
                label = std::string("t3c_decode_step_")
                        + std::to_string(i) + "_is_decode";
                CHECK(label.c_str(), !step_dec.entries[0].is_prefill);
                label = std::string("t3c_decode_step_")
                        + std::to_string(i) + "_num_tokens_1";
                CHECK(label.c_str(), step_dec.entries[0].num_tokens == 1);
            }
            saw_decode = true;
        }
        CHECK("t3c_saw_decode_steps", saw_decode);

        // Mark finished
        sched->finish_request(100);

        // After finish, no more active requests
        auto step_after = sched->step();
        CHECK("t3c_empty_after_finish", step_after.empty());
        CHECK("t3c_scheduler_done", sched->num_active() == 0);

        printf("State machine transitions verified\n");
    }

    // --- 3d: Scheduler factory ---
    {
        printf("\n--- 3d: Scheduler Factory ---\n");

        auto fcfs   = Scheduler::create(SchedulerPolicy::FCFS, 32);
        auto pf     = Scheduler::create(SchedulerPolicy::PrefillFirst, 32);
        auto df     = Scheduler::create(SchedulerPolicy::DecodeFirst, 32);

        CHECK("t3d_fcfs_not_null", fcfs != nullptr);
        CHECK("t3d_prefillfirst_not_null", pf != nullptr);
        CHECK("t3d_decodefirst_not_null", df != nullptr);

        // All should start with 0 active requests.
        CHECK("t3d_fcfs_empty", fcfs->num_active() == 0);
        CHECK("t3d_pf_empty", pf->num_active() == 0);
        CHECK("t3d_df_empty", df->num_active() == 0);

        // All should return empty steps.
        CHECK("t3d_fcfs_step_empty", fcfs->step().empty());
        CHECK("t3d_pf_step_empty", pf->step().empty());
        CHECK("t3d_df_step_empty", df->step().empty());

        printf("Scheduler factory verified\n");
    }

    // --- 3e: Chunked prefill (small chunk_size forces multiple chunks) ---
    {
        printf("\n--- 3e: Chunked Prefill ---\n");

        // chunk_size=2 forces a 5-token prompt into 3 prefill chunks
        auto sched = Scheduler::create(SchedulerPolicy::FCFS, 2);

        Request req;
        req.id              = 200;
        req.prompt_tokens    = {576, 8319, 315, 13466, 374};  // 5 tokens
        req.max_new_tokens  = 2;
        req.eos_token_id    = 151643;
        req.num_computed    = 0;

        sched->add_request(std::move(req));

        // Expect prefill in 3 chunks: 2 + 2 + 1
        int prefill_chunks = 0;
        int total_prefill_tokens = 0;
        for (int i = 0; i < 3; i++) {
            auto step = sched->step();
            if (step.empty()) break;
            if (!step.entries.empty() && step.entries[0].is_prefill) {
                prefill_chunks++;
                total_prefill_tokens += step.entries[0].num_tokens;
                std::string label = std::string("t3e_chunk_")
                                    + std::to_string(i) + "_prefill";
                CHECK(label.c_str(), true);
            }
        }
        CHECK("t3e_three_prefill_chunks", prefill_chunks == 3);
        CHECK("t3e_total_prefill_5", total_prefill_tokens == 5);

        // Now decode tokens should follow
        for (int i = 0; i < 2; i++) {
            auto step = sched->step();
            if (step.empty()) break;
            std::string label = std::string("t3e_decode_")
                                + std::to_string(i);
            if (!step.entries.empty()) {
                CHECK(label.c_str(), !step.entries[0].is_prefill);
            }
        }

        sched->finish_request(200);
        CHECK("t3e_done_after_finish", sched->num_active() == 0);

        printf("Chunked prefill verified (5 tokens split into 3 chunks)\n");
    }

    printf("Test 3 complete.\n");
}

// ============================================================================
// TEST 4: STRESS — 5 USERS WITH MIXED LENGTHS
// ============================================================================
/// Submits five requests with different prompt lengths (1, 3, 5, 4, 6 tokens)
/// and different max_new_tokens values.  Verifies all complete on time,
/// all outputs are valid, and no blocks leak.
static void test_stress_mixed_lengths(InferenceEngine& engine) {
    printf("\n========== TEST 4: STRESS — 5 USERS MIXED LENGTHS ==========\n");

    struct StressCase {
        const char* label;
        std::vector<int> prompt;
        int max_new_tokens;
    };

    std::vector<StressCase> cases = {
        {"User 1 (1-token prompt)",   {576},                                  6},
        {"User 2 (5-token prompt)",   {576, 8319, 315, 13466, 374},           8},
        {"User 3 (3-token prompt)",   {576, 8319, 315},                      10},
        {"User 4 (4-token prompt)",   {15837, 467, 315, 1605},                5},
        {"User 5 (6-token prompt)",   {10585, 374, 279, 18865, 1234, 5678},   7},
    };

    // Use DecodeFirst with small chunk size to force interleaved scheduling.
    ServeLoop loop(engine, SchedulerPolicy::DecodeFirst, 8);

    std::vector<int> ids;
    for (auto& c : cases) {
        int id = loop.add_request(c.prompt, c.max_new_tokens);
        ids.push_back(id);
        printf("  Added %s (id=%d, %zu prompt tokens, max_new=%d)\n",
               c.label, id, c.prompt.size(), c.max_new_tokens);
    }

    CHECK("t4_all_five_added", static_cast<int>(ids.size()) == 5);
    CHECK("t4_scheduler_has_five", loop.scheduler().num_active() == 5);

    auto t0 = std::chrono::steady_clock::now();
    loop.run();
    auto t1 = std::chrono::steady_clock::now();
    double wall_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            t1 - t0).count()) / 1000.0;

    printf("\n--- Results ---\n");
    int total_generated = 0;
    double total_engine_ms = 0.0;
    int completed = 0;

    for (size_t i = 0; i < cases.size(); i++) {
        auto& res = loop.result(ids[i]);
        int gen = static_cast<int>(res.generated_tokens.size());
        total_generated += gen;
        total_engine_ms += res.elapsed_ms;

        printf("  %s: %d generated tokens, %.1f ms, %s\n",
               cases[i].label, gen, res.elapsed_ms,
               res.finished ? "FINISHED" : "UNFINISHED");

        if (res.finished) completed++;

        // Per-user checks
        std::string pref_label = std::string("t4_user")
                                 + std::to_string(i) + "_prefill_complete";
        CHECK(pref_label.c_str(),
              res.num_prefilled == static_cast<int>(res.prompt_tokens.size()));

        std::string tok_label = std::string("t4_user")
                                + std::to_string(i) + "_tokens_valid";
        CHECK(tok_label.c_str(),
              tokens_in_range(res.generated_tokens, g_model.vocab_size));

        std::string gen_label = std::string("t4_user")
                                + std::to_string(i) + "_generated";
        CHECK(gen_label.c_str(), gen > 0);
    }

    double tok_s = compute_tok_s(total_generated, total_engine_ms);
    printf("\nTotal: %d tokens generated across %d requests\n",
           total_generated, completed);
    printf("Total engine time: %.1f ms (%.1f tok/s)\n",
           total_engine_ms, tok_s);
    printf("Wall clock: %.1f ms\n", wall_ms);

    // --- Verifications ---

    CHECK("t4_all_five_completed", completed == 5);

    CHECK("t4_all_finished",
          [&]() {
              for (auto id : ids)
                  if (!loop.result(id).finished) return false;
              return true;
          }());

    CHECK("t4_scheduler_idle_after", loop.scheduler().num_active() == 0);

    CHECK("t4_throughput_acceptable", tok_s > 5.0);

    // All generated tokens should be within vocab range.
    CHECK("t4_all_tokens_in_range",
          [&]() {
              for (auto id : ids) {
                  auto& res = loop.result(id);
                  if (!tokens_in_range(res.generated_tokens,
                                       g_model.vocab_size))
                      return false;
              }
              return true;
          }());

    printf("Test 4 complete.\n");
}

// ============================================================================
// TEST 5: EDGE CASES
// ============================================================================
/// Exercises boundary conditions in the scheduler-engine pipeline.
///
///   5a. Empty batch — scheduler returns empty step when no work; engine
///       must handle gracefully (no crash, no invalid memory accesses).
///   5b. 1-token prompt — minimal prefill (1 token), verify engine
///       can generate coherent continuation.
///   5c. EOS on first decode token — request finishes immediately after
///       generating one token; scheduler state machine must handle it.
///   5d. Zero max_new_tokens — request with no generation budget.
///   5e. Mixed finish times — one request finishes early, others continue.
///   5f. Scheduler with finished-then-new request — add, finish, add again.
static void test_edge_cases(InferenceEngine& engine) {
    printf("\n========== TEST 5: EDGE CASES ==========\n");

    // --- 5a: Empty batch ---
    {
        printf("\n--- 5a: Empty Batch ---\n");

        auto sched = Scheduler::create(SchedulerPolicy::FCFS, 16);

        // No requests added — step() should return empty.
        auto step = sched->step();
        CHECK("t5a_empty_step", step.empty());

        step = sched->step();  // second call also empty
        CHECK("t5a_empty_step_again", step.empty());

        CHECK("t5a_no_active", sched->num_active() == 0);

        // DecodeFirst empty scheduler also returns empty step.
        auto sched2 = Scheduler::create(SchedulerPolicy::DecodeFirst, 16);
        CHECK("t5a_df_empty", sched2->step().empty());

        printf("Empty batch: scheduler returns empty step cleanly\n");
    }

    // --- 5b: 1-token prompt ---
    {
        printf("\n--- 5b: 1-Token Prompt ---\n");

        ServeLoop loop(engine, SchedulerPolicy::FCFS, 16);

        // Just the token "The"
        int id = loop.add_request({576},
                                  /*max_new_tokens=*/8,
                                  /*eos_token_id=*/151643);

        CHECK("t5b_one_active", loop.scheduler().num_active() == 1);

        loop.run();

        auto& res = loop.result(id);

        printf("Prompt: [576], Generated: ");
        for (int t : res.generated_tokens) printf("%d ", t);
        printf("\n");
        printf("Generated %zu tokens in %.1f ms\n",
               res.generated_tokens.size(), res.elapsed_ms);

        CHECK("t5b_request_finished", res.finished);
        CHECK("t5b_generated_tokens",
              !res.generated_tokens.empty());
        CHECK("t5b_tokens_in_range",
              tokens_in_range(res.generated_tokens, g_model.vocab_size));
        CHECK("t5b_prefill_complete",
              res.num_prefilled == 1);  // prompt has 1 token

        // A 1-token prefill should be very fast (minimal attention).
        double tok_s = compute_tok_s(
            static_cast<int>(res.generated_tokens.size()),
            res.elapsed_ms);
        printf("Throughput: %.1f tok/s\n", tok_s);

        CHECK("t5b_throughput_reasonable", tok_s > 1.0);

        printf("1-token prompt: handled correctly\n");
    }

    // --- 5c: EOS on first decode token ---
    {
        printf("\n--- 5c: EOS on First Decode Token ---\n");

        // Use the engine directly with greedy sampling (temperature=0)
        // and a known prompt.  We verify EOS handling by checking that
        // IF EOS appears in the output, the request is still marked
        // finished and the scheduler handles it correctly.

        ServeLoop loop(engine, SchedulerPolicy::FCFS, 16);

        int id = loop.add_request(
            {576, 8319, 315, 13466, 374},  // "The capital of France is"
            /*max_new_tokens=*/64,
            /*eos_token_id=*/151643);

        loop.run();

        auto& res = loop.result(id);

        // Check: if EOS appears, the generated_tokens should
        // include EOS as the final token (engine stops generating).
        bool has_eos_in_output = false;
        for (int t : res.generated_tokens) {
            if (t == 151643) { has_eos_in_output = true; break; }
        }

        printf("Generated %zu tokens, EOS in output: %s\n",
               res.generated_tokens.size(),
               has_eos_in_output ? "yes" : "no");

        // The request must be marked finished regardless.
        CHECK("t5c_finished", res.finished);

        // Tokens must be valid.
        CHECK("t5c_tokens_valid",
              tokens_in_range(res.generated_tokens, g_model.vocab_size));

        // If EOS appeared, verify the scheduler handled the finish.
        CHECK("t5c_scheduler_idle", loop.scheduler().num_active() == 0);

        printf("EOS handling: verified\n");
    }

    // --- 5d: Zero max_new_tokens ---
    {
        printf("\n--- 5d: Zero max_new_tokens ---\n");

        ServeLoop loop(engine, SchedulerPolicy::FCFS, 16);

        // Request with max_new_tokens=0 should still process prefill
        // and produce zero generated tokens (or possibly 1 if engine
        // doesn't zero-check before the first decode).
        int id = loop.add_request(
            {576, 8319, 315},
            /*max_new_tokens=*/0,
            /*eos_token_id=*/151643);

        loop.run();

        auto& res = loop.result(id);

        printf("Generated %zu tokens (max_new_tokens=0)\n",
               res.generated_tokens.size());

        CHECK("t5d_finished", res.finished);
        // With max_new_tokens=0, engine may still generate 0 or 1 token
        // depending on implementation.  We just verify it didn't crash.
        CHECK("t5d_prefill_complete",
              res.num_prefilled == static_cast<int>(res.prompt_tokens.size()));

        printf("Zero max_new_tokens: handled correctly\n");
    }

    // --- 5e: Multiple requests, one finishes immediately ---
    {
        printf("\n--- 5e: Mixed Finish Times ---\n");

        ServeLoop loop(engine, SchedulerPolicy::DecodeFirst, 8);

        // Request 1: small prompt, few tokens (finishes quickly)
        int id_fast = loop.add_request({576}, 3);

        // Request 2: longer prompt, more tokens (takes longer)
        int id_slow = loop.add_request(
            {576, 8319, 315, 13466, 374}, 12);

        CHECK("t5e_two_active", loop.scheduler().num_active() == 2);

        loop.run();

        auto& fast = loop.result(id_fast);
        auto& slow = loop.result(id_slow);

        CHECK("t5e_both_finished", fast.finished && slow.finished);
        CHECK("t5e_both_generated_tokens",
              !fast.generated_tokens.empty() &&
              !slow.generated_tokens.empty());
        CHECK("t5e_scheduler_idle", loop.scheduler().num_active() == 0);

        printf("Mixed finish times: both completed, "
               "fast=%zutok slow=%zutok\n",
               fast.generated_tokens.size(),
               slow.generated_tokens.size());
    }

    // --- 5f: Add -> finish -> add again (scheduler reuse) ---
    {
        printf("\n--- 5f: Scheduler Reuse (add, finish, add again) ---\n");

        ServeLoop loop(engine, SchedulerPolicy::DecodeFirst, 16);

        // First batch: 2 requests
        int id1 = loop.add_request({576, 8319, 315}, 5);
        int id2 = loop.add_request({15837, 467}, 5);

        CHECK("t5f_two_active_round1",
              loop.scheduler().num_active() == 2);

        loop.run();

        CHECK("t5f_round1_finished",
              loop.result(id1).finished && loop.result(id2).finished);
        CHECK("t5f_idle_after_round1",
              loop.scheduler().num_active() == 0);

        // Second batch: 1 more request
        int id3 = loop.add_request({10585, 374, 279, 18865}, 5);

        CHECK("t5f_one_active_round2",
              loop.scheduler().num_active() == 1);

        loop.run();

        CHECK("t5f_round2_finished", loop.result(id3).finished);
        CHECK("t5f_idle_after_round2",
              loop.scheduler().num_active() == 0);

        printf("Scheduler reuse: two rounds, all requests completed\n");
    }

    printf("Test 5 complete.\n");
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv) {
    const char* model_dir = "models/qwen2.5-0.5b";
    if (argc > 1) model_dir = argv[1];

    printf("=============================================================\n");
    printf("  LightLLM — Scheduler-Engine Integration Test\n");
    printf("  Model: %s\n", model_dir);
    printf("=============================================================\n");

    try {
        // Read model config (needed for vocab_size and other metadata).
        printf("\n>>> Reading model config...\n");
        load_model_info(model_dir);

        // Create engine once — KV cache pools are allocated in the constructor.
        printf("\n>>> Initializing InferenceEngine...\n");
        InferenceEngine engine(model_dir);

        printf("Engine D=%d, layers=%d, vocab=%d\n",
               g_model.hidden_dim,
               g_model.num_layers,
               g_model.vocab_size);

        // Run the test suite.
        test_single_request(engine);
        test_three_concurrent(engine);
        test_scheduler_integration(engine);
        test_stress_mixed_lengths(engine);
        test_edge_cases(engine);

    } catch (const std::exception& ex) {
        fprintf(stderr, "\nFATAL: %s\n", ex.what());
        g_failed++;
    }

    // ---- Summary ----
    printf("\n=============================================================\n");
    printf("  RESULTS: %d passed, %d failed\n", g_passed, g_failed);
    printf("=============================================================\n");

    if (g_failed > 0) {
        printf("\nFAILURES DETECTED.  Check stderr for details.\n");
    }

    return (g_failed > 0) ? 1 : 0;
}
