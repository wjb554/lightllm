/// LightLLM — High-throughput simulation with mixed-length requests.
/// Maximizes GPU memory utilization, leaves ~500 MB headroom.
/// Usage: bench_throughput [duration_sec=60] [arrival_rate=1.0]

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
#include <vector>

#include "lightllm/engine/engine.h"
#include "lightllm/engine/batch_loop.h"

using namespace lightllm::engine;

static double now_sec() {
    static auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
static std::string ms_str(double v) { char b[32]; snprintf(b, sizeof(b), "%.0f ms", v); return b; }
static double pct(std::vector<double>& v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min((int)(p/100.0*v.size()), (int)v.size()-1)];
}

int main(int argc, char** argv) {
    double duration_sec  = (argc > 1) ? std::atof(argv[1]) : 60.0;
    double arrival_rate  = (argc > 2) ? std::atof(argv[2]) : 1.0;

    // --- Prompt templates by length category ---
    struct PromptSet {
        std::vector<int> tokens;
        int max_new;
    };
    // Short: 1-2 token prompts
    std::vector<PromptSet> short_prompts = {
        {{576}, 8},                    // "The"
        {{15837}, 6},                  // "Weather"
        {{10585}, 7},                  // "I"
        {{785}, 5},                    // "What"
    };
    // Medium: 3-5 token prompts
    std::vector<PromptSet> med_prompts = {
        {{576, 8319, 315, 13466, 374}, 15},       // "The capital of France is"
        {{15837, 467, 315, 1605}, 12},             // "The weather today is"
        {{10585, 374, 279, 18865}, 14},            // "I think therefore I"
        {{785, 1791, 315, 4166}, 10},              // "What is the meaning"
    };
    // Long: 6-14 token prompts
    std::vector<PromptSet> long_prompts = {
        {{576, 374, 279, 3465, 320, 374, 279, 220, 285, 315, 24794}, 25},
        {{1339, 315, 220, 285, 315, 24794, 576, 8319, 315, 13466, 374}, 30},
    };

    // --- Memory config ---
    const int GPU_MB = 6 * 1024;       // RTX 2060 6 GB
    const int MODEL_MB = 2100;          // 494M params × 4 bytes + overhead
    const int RESERVED_MB = 500;        // 500 MB headroom
    const int KV_CACHE_MB = GPU_MB - MODEL_MB - RESERVED_MB;  // ~3500 MB
    const int CHUNK_SIZE = 16;
    const int MAX_BATCH_TOKENS = 512;   // Larger budget for high throughput

    printf("=============================================================\n");
    printf("  LightLLM — High-Throughput Simulation\n");
    printf("  Duration: %.0f s | Arrival: %.2f req/s\n", duration_sec, arrival_rate);
    printf("  GPU: %d MB | Model: %d MB | Reserved: %d MB | KV Cache: %d MB\n",
           GPU_MB, MODEL_MB, RESERVED_MB, KV_CACHE_MB);
    printf("  Short/Med/Long ratio: 50%%/35%%/15%%\n");
    printf("=============================================================\n\n");

    // --- Pre-generate arrivals with mixed lengths ---
    std::mt19937 rng(42);
    std::exponential_distribution<double> inter_arr(arrival_rate);
    std::uniform_real_distribution<double> len_rand(0.0, 1.0);
    std::uniform_int_distribution<int> short_pick(0, (int)short_prompts.size()-1);
    std::uniform_int_distribution<int> med_pick(0, (int)med_prompts.size()-1);
    std::uniform_int_distribution<int> long_pick(0, (int)long_prompts.size()-1);

    struct Arrival { double t; std::vector<int> prompt; int max_new; int cat; };
    std::deque<Arrival> arrivals;
    double t = 0;
    int n_short = 0, n_med = 0, n_long = 0;
    while (t < duration_sec) {
        t += inter_arr(rng);
        if (t >= duration_sec) break;
        double r = len_rand(rng);
        if (r < 0.50) {
            auto& ps = short_prompts[short_pick(rng)];
            arrivals.push_back({t, ps.tokens, ps.max_new, 0}); n_short++;
        } else if (r < 0.85) {
            auto& ps = med_prompts[med_pick(rng)];
            arrivals.push_back({t, ps.tokens, ps.max_new, 1}); n_med++;
        } else {
            auto& ps = long_prompts[long_pick(rng)];
            arrivals.push_back({t, ps.tokens, ps.max_new, 2}); n_long++;
        }
    }
    printf("Pre-generated: %d short + %d med + %d long = %zu total\n\n",
           n_short, n_med, n_long, arrivals.size());

    // --- Init engine with maxed-out KV cache ---
    printf("Loading model with %d MB KV cache pool...\n", KV_CACHE_MB);
    EngineServer engine("models/qwen2.5-0.5b", /*max_seq_len=*/0,
                        MAX_BATCH_TOKENS, KV_CACHE_MB);
    printf("KV pool: %d blocks/layer × %d layers = %.0f MB\n",
           engine.num_blocks(), engine.num_layers(),
           (double)engine.num_blocks() * 16 * 2 * 64 * 2 * 4
               * engine.num_layers() / 1048576.0);

    BatchMainLoop batch(engine, SchedulerPolicy::DecodeFirst,
                        CHUNK_SIZE, MAX_BATCH_TOKENS);

    struct Tracked { int loop_id; double arrival; int cat; };
    std::vector<Tracked> tracked;
    double sim_start = now_sec();
    double next_report = 10.0;
    int total_submitted = 0;

    printf("Time(s) | Active | KV%%  | Done  | Tok/s | AvgBatch | P50_TTFT | P50_TPOT\n");
    printf("--------|--------|------|-------|-------|----------|-----------|---------\n");

    while (true) {
        double elapsed = now_sec() - sim_start;

        // Inject arrivals
        while (!arrivals.empty() && arrivals.front().t <= elapsed) {
            auto a = arrivals.front(); arrivals.pop_front();
            int id = batch.submit(a.prompt, a.max_new, 151643);
            tracked.push_back({id, a.t, a.cat});
            total_submitted++;
        }

        // Sleep if idle and waiting for next arrival
        if (!batch.has_active() && !arrivals.empty()) {
            double wait = arrivals.front().t - elapsed;
            if (wait > 0.001) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(std::max(1, (int)(wait*1000))));
                continue;
            }
        }

        batch.step();

        // Periodic report
        if (elapsed >= next_report) {
            // KV utilization: estimate from active requests' block usage
            int done = 0, short_done = 0, med_done = 0, long_done = 0;
            for (auto& tr : tracked) {
                auto& m = batch.all_metrics()[tr.loop_id];
                if (m.finished) {
                    done++;
                    if (tr.cat == 0) short_done++;
                    else if (tr.cat == 1) med_done++;
                    else long_done++;
                }
            }

            int used = engine.used_blocks();
            int kv_pct = engine.num_blocks() > 0
                ? (used * 100 / engine.num_blocks()) : 0;

            printf("%7.0f | %6d | %3d%%  | %5d | %5.0f | %8.1f | %9s | %8s\n",
                   elapsed,
                   batch.active_count(),
                   kv_pct,
                   done,
                   batch.total_tokens_generated() / std::max(elapsed, 1.0),
                   batch.total_steps() > 0
                       ? (double)batch.total_tokens_generated() / batch.total_steps() : 0,
                   ms_str(batch.p50_ttft_ms()).c_str(),
                   ms_str(batch.p50_tpot_ms()).c_str());

            next_report += 10.0;
        }

        if (arrivals.empty() && !batch.has_active()) break;
    }

    double sim_elapsed = now_sec() - sim_start;

    // --- Final report ---
    printf("\n=============================================================\n");
    printf("  FINAL REPORT\n");
    printf("=============================================================\n");
    printf("  Wall time:       %.1f s\n", sim_elapsed);
    printf("  Total submitted: %d\n", total_submitted);

    int finished = 0, short_fin = 0, med_fin = 0, long_fin = 0;
    std::vector<double> ttfts, tpots, latencies, outlens;
    for (auto& tr : tracked) {
        auto& m = batch.all_metrics()[tr.loop_id];
        if (m.finished) {
            finished++;
            if (tr.cat == 0) short_fin++;
            else if (tr.cat == 1) med_fin++;
            else long_fin++;
            ttfts.push_back(m.ttft_ms());
            tpots.push_back(m.tpot_ms());
            latencies.push_back(m.latency_ms());
            outlens.push_back(m.output_len);
        }
    }

    printf("  Finished:        %d (S:%d M:%d L:%d)\n", finished, short_fin, med_fin, long_fin);
    printf("  Throughput:      %.1f tok/s (%.2f req/s)\n",
           sim_elapsed > 0 ? batch.total_tokens_generated() / sim_elapsed : 0,
           sim_elapsed > 0 ? finished / sim_elapsed : 0);

    printf("\n  -- Latency (ms) --\n");
    printf("  %-12s %8s %8s %8s %8s %8s\n", "Metric", "P50", "P90", "P95", "P99", "Avg");
    printf("  %-12s %8s %8s %8s %8s %8s\n", "TTFT",
           ms_str(pct(ttfts, 50)).c_str(), ms_str(pct(ttfts, 90)).c_str(),
           ms_str(pct(ttfts, 95)).c_str(), ms_str(pct(ttfts, 99)).c_str(),
           ms_str(ttfts.empty() ? 0 : std::accumulate(ttfts.begin(), ttfts.end(), 0.0)/ttfts.size()).c_str());
    printf("  %-12s %8s %8s %8s %8s %8s\n", "TPOT",
           ms_str(pct(tpots, 50)).c_str(), ms_str(pct(tpots, 90)).c_str(),
           ms_str(pct(tpots, 95)).c_str(), ms_str(pct(tpots, 99)).c_str(),
           ms_str(tpots.empty() ? 0 : std::accumulate(tpots.begin(), tpots.end(), 0.0)/tpots.size()).c_str());
    printf("  %-12s %8s %8s %8s %8s %8s\n", "Total",
           ms_str(pct(latencies, 50)).c_str(), ms_str(pct(latencies, 90)).c_str(),
           ms_str(pct(latencies, 95)).c_str(), ms_str(pct(latencies, 99)).c_str(),
           ms_str(latencies.empty() ? 0 : std::accumulate(latencies.begin(), latencies.end(), 0.0)/latencies.size()).c_str());

    std::sort(outlens.begin(), outlens.end());
    printf("\n  -- Output lengths --\n");
    printf("  Min: %.0f  P50: %.0f  P90: %.0f  Max: %.0f\n",
           pct(outlens, 0), pct(outlens, 50), pct(outlens, 90), pct(outlens, 100));

    printf("\n  -- KV Cache --\n");
    printf("  Pool size: %d blocks (%.0f MB) across %d layers\n",
           engine.num_blocks(),
           (double)engine.num_blocks() * 16 * 2 * 64 * 2 * 4
               * engine.num_layers() / 1048576.0,
           engine.num_layers());
    int peak_used = engine.num_blocks() - engine.free_blocks();
    printf("  Current utilization: %d/%d blocks (%.1f MB, %.1f%%)\n",
           engine.used_blocks(), engine.num_blocks(),
           (double)engine.used_blocks() * 16 * 2 * 64 * 2 * 4 / 1048576.0,
           engine.num_blocks() > 0 ? (engine.used_blocks() * 100.0 / engine.num_blocks()) : 0);

    printf("\n  Simulation complete.\n");
    return 0;
}
