/// bench_server.cpp — demonstrate the Scheduler-Engine integration.
///
/// Creates 3 users with different prompts, runs the DecodeFirst scheduler
/// and EngineServer in a continuous batching loop, and reports per-user output
/// and throughput metrics.
///
/// Usage:
///   bench_server <model_dir> [max_seq_len] [max_batch_tokens]
///
/// Example:
///   bench_server ../models/Qwen2.5-0.5B-Instruct 2048 256

#include "lightllm/engine/scheduler.h"
#include "lightllm/engine/engine.h"
#include "lightllm/kv_cache/block_allocator.h"
#include "lightllm/tensor.h"
#include "lightllm/model/model_config.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lightllm::engine;

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: bench_server <model_dir> [max_seq_len] "
                "[max_batch_tokens]\n");
        return 1;
    }

    std::string model_dir = argv[1];
    int max_seq_len       = (argc >= 3) ? std::atoi(argv[2]) : 2048;
    int max_batch_tokens  = (argc >= 4) ? std::atoi(argv[3]) : 256;

    printf("============================================================\n");
    printf("  LightLLM Scheduler-Engine Benchmark\n");
    printf("============================================================\n");
    printf("  Model dir:      %s\n", model_dir.c_str());
    printf("  Max seq len:    %d\n", max_seq_len);
    printf("  Max batch tok:  %d\n", max_batch_tokens);
    printf("  Scheduler:      DecodeFirst (chunk_size=16)\n");
    printf("============================================================\n\n");

    // ------------------------------------------------------------------
    // 1.  INITIALIZATION
    // ------------------------------------------------------------------
    printf("[init] Loading model...\n");
    EngineServer engine(model_dir, max_seq_len, max_batch_tokens);
    printf("[init] Model loaded. Hidden dim=%d, vocab=%d, layers=%d\n",
           engine.hidden_dim(), engine.vocab_size(), engine.num_layers());

    auto sched = Scheduler::create(SchedulerPolicy::DecodeFirst, 16);
    std::unordered_map<int, std::unique_ptr<RequestState>> states;

    // ------------------------------------------------------------------
    // 2.  ADD 3 TEST USERS
    // ------------------------------------------------------------------

    int next_id = 0;

    // User 1: short prompt, moderate generation
    {
        Request req;
        req.id = next_id++;
        req.prompt_tokens = {1, 2, 3, 4, 5};
        req.max_new_tokens = 15;
        req.eos_token_id = 151643;

        int uid = req.id;
        size_t plen = req.prompt_tokens.size();
        int mnt = req.max_new_tokens;

        auto state = engine.create_request_state(req, engine.hidden_dim());
        states[uid] = std::move(state);
        sched->add_request(std::move(req));

        printf("[user %d] added: prompt_len=%zu, max_new=%d\n",
               uid, plen, mnt);
    }

    // User 2: very short prompt, short generation
    {
        Request req;
        req.id = next_id++;
        req.prompt_tokens = {10, 20, 30};
        req.max_new_tokens = 8;
        req.eos_token_id = 151643;

        int uid = req.id;
        size_t plen = req.prompt_tokens.size();
        int mnt = req.max_new_tokens;

        auto state = engine.create_request_state(req, engine.hidden_dim());
        states[uid] = std::move(state);
        sched->add_request(std::move(req));

        printf("[user %d] added: prompt_len=%zu, max_new=%d\n",
               uid, plen, mnt);
    }

    // User 3: longer prompt, short generation
    {
        Request req;
        req.id = next_id++;
        req.prompt_tokens = {100, 200, 300, 400, 500, 600, 700};
        req.max_new_tokens = 10;
        req.eos_token_id = 151643;

        int uid = req.id;
        size_t plen = req.prompt_tokens.size();
        int mnt = req.max_new_tokens;

        auto state = engine.create_request_state(req, engine.hidden_dim());
        states[uid] = std::move(state);
        sched->add_request(std::move(req));

        printf("[user %d] added: prompt_len=%zu, max_new=%d\n",
               uid, plen, mnt);
    }

    printf("[init] %d active requests\n\n", sched->num_active());

    // ------------------------------------------------------------------
    // 3.  SERVE LOOP
    // ------------------------------------------------------------------

    int total_prefill_tokens = 0;
    int total_decode_steps   = 0;
    int total_generated      = 0;
    int total_steps          = 0;

    // Per-user token tracking
    struct UserStats {
        int id;
        int prompt_len;
        int tokens_generated;
        bool finished;
    };
    std::vector<UserStats> user_stats;
    for (auto& [id, st] : states) {
        user_stats.push_back({id, (int)st->prompt_tokens.size(), 0, false});
    }

    auto t_start = std::chrono::steady_clock::now();

    printf("[serve] Starting continuous batching loop...\n\n");

    while (sched->num_active() > 0) {
        // ----- 3a.  Get next batch from scheduler -----
        ScheduleStep step = sched->step();

        if (step.empty()) {
            // No work available; could yield but for benchmark just continue
            continue;
        }

        total_steps++;

        // Count batch composition
        int n_prefill = 0, n_decode = 0, n_prefill_tok = 0;
        for (auto& e : step.entries) {
            if (e.is_prefill) { n_prefill++; n_prefill_tok += e.num_tokens; }
            else              { n_decode++; }
        }

        // ----- 3b.  Execute the batch on the engine -----
        auto sampled_tokens = engine.step(step, states);

        total_prefill_tokens += n_prefill_tok;
        total_decode_steps   += n_decode;

        // ----- 3c.  Process results -----
        for (auto& tok : sampled_tokens) {
            total_generated++;

            // Update per-user stats
            for (auto& us : user_stats) {
                if (us.id == tok.request_id) {
                    us.tokens_generated++;
                    if (tok.is_eos) us.finished = true;
                    break;
                }
            }

            if (tok.is_eos) {
                auto it = states.find(tok.request_id);
                if (it != states.end()) {
                    engine.release_request(*it->second);
                    states.erase(it);
                }
                sched->finish_request(tok.request_id);
            }
        }

        // Progress indicator every 10 steps
        if (total_steps % 10 == 0) {
            printf("  [step %3d] batch: %dP/%dD tokens, %d generated, "
                   "%d active\n",
                   total_steps, n_prefill_tok, n_decode,
                   total_generated, sched->num_active());
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();
    double elapsed_s = elapsed_ms / 1000.0;

    // ------------------------------------------------------------------
    // 4.  REPORT
    // ------------------------------------------------------------------

    printf("\n============================================================\n");
    printf("  RESULTS\n");
    printf("============================================================\n\n");

    printf("  Per-user output:\n");
    printf("  %-6s %12s %18s %10s\n",
           "User", "Prompt len", "Tokens generated", "Status");
    printf("  %-6s %12s %18s %10s\n",
           "------", "----------", "-----------------", "----------");
    for (auto& us : user_stats) {
        printf("  %-6d %12d %18d %10s\n",
               us.id, us.prompt_len, us.tokens_generated,
               us.finished ? "FINISHED" : "TIMEOUT");
    }

    printf("\n  Throughput:\n");
    printf("  %-30s %d\n",   "Total steps:",  total_steps);
    printf("  %-30s %d\n",   "Prefill tokens processed:", total_prefill_tokens);
    printf("  %-30s %d\n",   "Decode steps:",  total_decode_steps);
    printf("  %-30s %d\n",   "Tokens generated:", total_generated);
    printf("  %-30s %.2f s\n", "Elapsed time:", elapsed_s);
    if (elapsed_s > 0.0) {
        printf("  %-30s %.1f tok/s\n", "Generation throughput:",
               total_generated / elapsed_s);
        printf("  %-30s %.1f steps/s\n", "Steps per second:",
               total_steps / elapsed_s);
    }

    // Check for leaks: all states should be released
    if (!states.empty()) {
        printf("\n  WARNING: %zu unreleased RequestState(s) remaining\n",
               states.size());
        for (auto& [id, st] : states) {
            engine.release_request(*st);
        }
        states.clear();
    }

    printf("\n  Done.\n");
    return 0;
}
