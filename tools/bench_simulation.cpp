/// LightLLM — Multi-user simulation benchmark (v3 — continuous batching).
/// Simulates Poisson-arrival requests with BatchMainLoop for heterogeneous
/// batch execution.  Tracks per-request latency and per-step batch efficiency.
/// Usage: bench_simulation [duration_sec=60] [arrival_rate=0.5] [report_intv=10]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <numeric>
#include <random>
#include <thread>
#include <string>
#include <vector>

#include "lightllm/engine/engine.h"
#include "lightllm/engine/batch_loop.h"

using namespace lightllm::engine;

// ============================================================================
// Timing helpers
// ============================================================================

static double now_sec() {
    static auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static std::string ms_str(double v) {
    char b[32]; snprintf(b, sizeof(b), "%.0f ms", v); return b;
}

static double pct(std::vector<double>& v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    int i = std::min((int)(p / 100.0 * v.size()), (int)v.size() - 1);
    return v[i];
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    double duration_sec  = (argc > 1) ? std::atof(argv[1]) : 60.0;
    double arrival_rate  = (argc > 2) ? std::atof(argv[2]) : 0.5;
    int    report_intv   = (argc > 3) ? std::atoi(argv[3]) : 10;

    // --- Prompt templates ---
    std::vector<std::vector<int>> prompts = {
        {576, 8319, 315, 13466, 374},          // "The capital of France is"
        {15837, 467, 315, 1605},                // "The weather today is"
        {10585, 374, 279, 18865},               // "I think therefore I"
        {785, 1791, 315, 4166},                 // "What is the meaning"
        {576, 374, 279, 3465, 320},             // "The future of AI"
    };

    const int CHUNK_SIZE = 16;
    const int MAX_BATCH_TOKENS = 256;

    printf("=============================================================\n");
    printf("  LightLLM — Multi-User Simulation Benchmark (cont. batching)\n");
    printf("  Duration: %.0f s | Arrival: %.2f req/s | Prompt pool: %zu\n",
           duration_sec, arrival_rate, prompts.size());
    printf("  Chunk size: %d | Max batch tokens: %d\n",
           CHUNK_SIZE, MAX_BATCH_TOKENS);
    printf("=============================================================\n\n");

    // --- Pre-generate Poisson arrivals ---
    std::mt19937 rng(42);
    std::exponential_distribution<double> inter_arr(arrival_rate);
    std::uniform_int_distribution<int> pick_prompt(0, (int)prompts.size() - 1);
    std::uniform_int_distribution<int> pick_maxnew(6, 18);

    struct Arrival { double t; int pi; int max_new; };
    std::deque<Arrival> arrivals;
    double t = 0;
    while (t < duration_sec) {
        t += inter_arr(rng);
        if (t < duration_sec)
            arrivals.push_back({t, pick_prompt(rng), pick_maxnew(rng)});
    }
    printf("Pre-generated %zu requests\n\n", arrivals.size());

    // --- Init engine (batch-capable) ---
    printf("Loading model...\n");
    EngineServer engine("models/qwen2.5-0.5b", 0, MAX_BATCH_TOKENS);

    // --- Init batch loop ---
    BatchMainLoop batch_loop(engine, SchedulerPolicy::DecodeFirst,
                             CHUNK_SIZE, MAX_BATCH_TOKENS);

    struct TrackedReq {
        int loop_id;
        double arrival_time;
    };
    std::vector<TrackedReq> tracked;

    double sim_start = now_sec();
    double next_report = report_intv;
    int total_submitted = 0;

    printf("Time(s) | Active | Done  | Tok/s | Steps/s | AvgBatch | P50_TTFT | P50_TPOT | P50_Lat\n");
    printf("--------|--------|-------|-------|---------|----------|-----------|----------|---------\n");

    // --- Main loop: inject arrivals + step batch ---
    int prev_steps = 0;
    int prev_tokens = 0;

    while (true) {
        double elapsed = now_sec() - sim_start;

        // Inject new arrivals at their scheduled times.
        while (!arrivals.empty() && arrivals.front().t <= elapsed) {
            auto a = arrivals.front(); arrivals.pop_front();
            int loop_id = batch_loop.submit(prompts[a.pi], a.max_new, 151643);
            tracked.push_back({loop_id, a.t});
            total_submitted++;
        }

        // If idle and more arrivals are coming, sleep until the next one.
        if (!batch_loop.has_active() && !arrivals.empty()) {
            double wait_s = arrivals.front().t - elapsed;
            if (wait_s > 0.001) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(std::max(1, (int)(wait_s * 1000))));
                continue;
            }
        }

        // Execute one batch step.
        batch_loop.step();

        // Periodic report.
        if (elapsed >= next_report && elapsed > 0) {
            int done_now = 0;
            for (auto& m : batch_loop.all_metrics())
                if (m.finished) done_now++;

            int cur_steps = batch_loop.total_steps();
            int cur_tokens = batch_loop.total_tokens_generated();
            double interval_s = report_intv;
            double tok_s = (interval_s > 0) ? (cur_tokens - prev_tokens) / interval_s : 0;
            double steps_s = (interval_s > 0) ? (cur_steps - prev_steps) / interval_s : 0;

            double avg_batch = 0.0;
            int steps_in_interval = cur_steps - prev_steps;
            if (steps_in_interval > 0) {
                int total_batch_tokens = 0;
                for (int i = prev_steps; i < cur_steps && i < (int)batch_loop.step_metrics().size(); i++) {
                    total_batch_tokens += batch_loop.step_metrics()[i].tokens_processed;
                }
                avg_batch = (double)total_batch_tokens / steps_in_interval;
            }

            printf("%7.0f | %6d | %5d | %5.0f | %7.0f | %8.1f | %9s | %8s | %8s\n",
                   elapsed,
                   batch_loop.active_count(),
                   done_now,
                   tok_s,
                   steps_s,
                   avg_batch,
                   ms_str(batch_loop.p50_ttft_ms()).c_str(),
                   ms_str(batch_loop.p50_tpot_ms()).c_str(),
                   ms_str(batch_loop.p50_latency_ms()).c_str());

            next_report += report_intv;
            prev_steps = cur_steps;
            prev_tokens = cur_tokens;
        }

        // Exit condition: no more arrivals, no active requests.
        if (arrivals.empty() && !batch_loop.has_active()) break;
    }

    double sim_elapsed = now_sec() - sim_start;

    // ========================================================================
    // FINAL REPORT — Latency
    // ========================================================================

    printf("\n=============================================================\n");
    printf("  FINAL REPORT\n");
    printf("=============================================================\n");
    printf("  Wall time:            %.1f s\n", sim_elapsed);
    printf("  Total submitted:      %d\n", total_submitted);
    printf("  Total steps:          %d\n", batch_loop.total_steps());
    printf("  Total tokens gen'd:   %d\n", batch_loop.total_tokens_generated());
    printf("  Throughput:           %.1f tok/s\n",
           sim_elapsed > 0 ? batch_loop.total_tokens_generated() / sim_elapsed : 0);
    printf("  Throughput (req/s):   %.3f req/s\n",
           sim_elapsed > 0 ? total_submitted / sim_elapsed : 0);
    printf("  Avg steps/sec:        %.1f\n",
           sim_elapsed > 0 ? batch_loop.total_steps() / sim_elapsed : 0);

    // Collect per-request latency.
    std::vector<double> ttfts, tpots, latencies, outlens;
    int total_generated = 0;
    int finished_count = 0;

    for (auto& tr : tracked) {
        try {
            const auto& m = batch_loop.metrics(tr.loop_id);
            if (m.finished) {
                double ttft_s = m.first_token_time_sec - m.arrival_time_sec;
                double lat_s  = m.finish_time_sec - m.arrival_time_sec;
                double tpot_s = (m.output_len > 0)
                    ? (m.finish_time_sec - m.first_token_time_sec) / m.output_len
                    : 0.0;

                ttfts.push_back(ttft_s * 1000.0);
                tpots.push_back(tpot_s * 1000.0);
                latencies.push_back(lat_s * 1000.0);
                outlens.push_back(m.output_len);
                total_generated += m.output_len;
                finished_count++;
            }
        } catch (...) {}
    }

    printf("\n  Finished requests:    %d / %d\n", finished_count, total_submitted);

    printf("\n  -- Latency (ms) --\n");
    printf("  %-12s %8s %8s %8s %8s %8s\n",
           "Metric", "P50", "P90", "P95", "P99", "Avg");
    printf("  %-12s %8s %8s %8s %8s %8s\n", "TTFT",
           ms_str(pct(ttfts, 50)).c_str(), ms_str(pct(ttfts, 90)).c_str(),
           ms_str(pct(ttfts, 95)).c_str(), ms_str(pct(ttfts, 99)).c_str(),
           ms_str(ttfts.empty() ? 0 : std::accumulate(ttfts.begin(), ttfts.end(), 0.0) / ttfts.size()).c_str());
    printf("  %-12s %8s %8s %8s %8s %8s\n", "TPOT",
           ms_str(pct(tpots, 50)).c_str(), ms_str(pct(tpots, 90)).c_str(),
           ms_str(pct(tpots, 95)).c_str(), ms_str(pct(tpots, 99)).c_str(),
           ms_str(tpots.empty() ? 0 : std::accumulate(tpots.begin(), tpots.end(), 0.0) / tpots.size()).c_str());
    printf("  %-12s %8s %8s %8s %8s %8s\n", "Total",
           ms_str(pct(latencies, 50)).c_str(), ms_str(pct(latencies, 90)).c_str(),
           ms_str(pct(latencies, 95)).c_str(), ms_str(pct(latencies, 99)).c_str(),
           ms_str(latencies.empty() ? 0 : std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size()).c_str());

    // Output length distribution.
    std::sort(outlens.begin(), outlens.end());
    printf("\n  -- Output tokens per request --\n");
    printf("  Total: %d | Avg: %.1f | Min: %.0f | P50: %.0f | Max: %.0f\n",
           total_generated,
           outlens.empty() ? 0 : std::accumulate(outlens.begin(), outlens.end(), 0.0) / outlens.size(),
           pct(outlens, 0), pct(outlens, 50), pct(outlens, 100));

    // ========================================================================
    // BATCH EFFICIENCY SECTION
    // ========================================================================
    {
        const auto& steps = batch_loop.step_metrics();
        int n_steps = (int)steps.size();

        if (n_steps > 0) {
            std::vector<double> batch_tokens, batch_reqs;
            int total_prefill_tokens = 0, total_decode_tokens = 0;
            int peak_active = 0, peak_tokens = 0;

            for (auto& sm : steps) {
                batch_tokens.push_back(sm.tokens_processed);
                batch_reqs.push_back(sm.num_requests);
                total_prefill_tokens += sm.prefill_tokens;
                total_decode_tokens += sm.decode_tokens;
                if (sm.active_after > peak_active) peak_active = sm.active_after;
                if (sm.tokens_processed > peak_tokens) peak_tokens = sm.tokens_processed;
            }

            printf("\n  -- BATCH EFFICIENCY --\n");
            printf("  Total steps:          %d\n", n_steps);
            printf("  Avg tokens/step:      %.1f\n",
                   std::accumulate(batch_tokens.begin(), batch_tokens.end(), 0.0) / n_steps);
            printf("  Avg requests/step:    %.1f\n",
                   std::accumulate(batch_reqs.begin(), batch_reqs.end(), 0.0) / n_steps);
            printf("  Peak tokens/step:     %d\n", peak_tokens);
            printf("  Peak active requests: %d\n", peak_active);

            printf("  Prefill/decode ratio: %.2f (prefill: %d, decode: %d)\n",
                   total_decode_tokens > 0
                       ? (double)total_prefill_tokens / total_decode_tokens
                       : 0.0,
                   total_prefill_tokens, total_decode_tokens);

            printf("  Token distribution:\n");
            printf("    P50 tokens/step: %.0f\n", pct(batch_tokens, 50));
            printf("    P90 tokens/step: %.0f\n", pct(batch_tokens, 90));
            printf("    Max tokens/step: %.0f\n", pct(batch_tokens, 100));

            printf("  Request distribution:\n");
            printf("    P50 reqs/step:   %.0f\n", pct(batch_reqs, 50));
            printf("    P90 reqs/step:   %.0f\n", pct(batch_reqs, 90));
            printf("    Max reqs/step:   %.0f\n", pct(batch_reqs, 100));
        }
    }

    // Arrival rate analysis.
    printf("\n  -- Arrival Analysis --\n");
    printf("  Target rate:    %.2f req/s\n", arrival_rate);
    printf("  Actual rate:    %.3f req/s\n", total_submitted / duration_sec);
    double queue_time = sim_elapsed - duration_sec;
    printf("  Drain time:     %.1f s (processing queue after arrival window ends)\n",
           queue_time > 0 ? queue_time : 0.0);

    printf("\n  Simulation complete.\n");
    return 0;
}
