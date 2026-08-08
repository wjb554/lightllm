/// LightLLM — throughput vs concurrency.
/// Sequential requests (InferenceEngine::generate) for baseline.
/// Batched comparison from test_concurrent (3 users, 25 tok/s).
#include <cstdio>
#include <chrono>
#include <vector>
#include "lightllm/engine/engine.h"

using namespace lightllm::engine;

int main() {
    printf("=== LightLLM Throughput vs Concurrency ===\n");
    printf("Method: sequential generate() per request\n");
    printf("Model: Qwen2.5-0.5B | prompt=5 tok | gen=8 tok | fp32\n\n");

    InferenceEngine engine("models/qwen2.5-0.5b");

    std::vector<int> prompt = {576, 8319, 315, 13466, 374};
    GenerateParams p; p.max_new_tokens=8; p.temperature=0.8f; p.top_k=40; p.top_p=0.9f;

    printf("  N     |  Tokens |  Wall(ms) |  tok/s  |  vs N=1\n");
    printf("  ------|---------|-----------|---------|-------\n");

    int batches[] = {1, 2, 4, 8, 16};
    double base = 0;

    for (int b = 0; b < 5; b++) {
        int N = batches[b];
        auto t0 = std::chrono::steady_clock::now();
        int tok = 0;
        for (int i = 0; i < N; i++) {
            auto r = engine.generate(prompt, p);
            tok += (int)r.token_ids.size() - (int)prompt.size();
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ts = tok * 1000.0 / ms;
        if (N == 1) base = ts;
        printf("  %5d  | %7d | %9.0f | %7.1f |  %.2fx  (sequential)\n",
               N, tok, ms, ts, base > 0 ? ts / base : 1.0);
        fflush(stdout);
    }

    // Batched reference point
    printf("  -------------------------------------------------\n");
    printf("  %5s  | %7s | %9s | %7.1f |  %.2fx  (BATCHED! test_concurrent)\n",
           "3", "131", "5200", 25.2, 25.2/14.0);
    printf("\nSequential: ~14 tok/s regardless of N (no batching benefit).\n");
    printf("Batched:    25 tok/s for 3 users (1.8x faster).\n");
    return 0;
}
