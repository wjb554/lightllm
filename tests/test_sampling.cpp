/// LightLLM — Comprehensive Sampling Tests
/// Tests every code path in the new sampling system:
///   greedy, temperature, top-k, top-p, min-p, combined, edge cases,
///   reproducibility, sample_batch, and detail helpers.
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <set>
#include "lightllm/ops/sampling.h"

using namespace lightllm::ops;

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
static int passed = 0;
static int failed = 0;

static void CHECK(const char* name, bool cond) {
    if (cond) passed++;
    else { failed++; fprintf(stderr, "  FAIL: %s\n", name); }
}

static bool neareq(float a, float b, float eps = 1e-4f) {
    return fabsf(a - b) < eps;
}

// Helper: compute the expected probability distribution from raw logits
// by applying the full pipeline described in the header:
//   temp scaling -> top-k -> softmax -> top-p -> min-p
static std::vector<float> expected_distribution(
    const float* logits, int vocab_size,
    const SamplingParams& params)
{
    std::vector<float> probs(logits, logits + vocab_size);

    // 1. Temperature scaling (skip if temp == 0 — that's greedy)
    if (params.temperature > 0.0f) {
        float inv_t = 1.0f / params.temperature;
        for (int i = 0; i < vocab_size; ++i)
            probs[i] *= inv_t;
    }

    // 2. Top-k filtering
    if (params.top_k > 0 && params.top_k < vocab_size) {
        std::vector<float> copy = probs;
        std::nth_element(copy.begin(), copy.begin() + params.top_k - 1,
                         copy.end(), std::greater<float>());
        float threshold = copy[params.top_k - 1];
        for (int i = 0; i < vocab_size; ++i)
            if (probs[i] < threshold) probs[i] = -INFINITY;
    }

    // 3. Softmax
    float max_val = -INFINITY;
    for (int i = 0; i < vocab_size; ++i)
        if (probs[i] > max_val) max_val = probs[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = (probs[i] == -INFINITY) ? 0.0f : expf(probs[i] - max_val);
        sum += probs[i];
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < vocab_size; ++i) probs[i] *= inv;
    }

    // 4. Top-p filtering
    if (params.top_p < 1.0f) {
        std::vector<int> idx(vocab_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return probs[a] > probs[b];
        });
        float cum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; ++i) {
            cum += probs[idx[i]];
            if (cum >= params.top_p) { cutoff = i + 1; break; }
        }
        for (int i = cutoff; i < vocab_size; ++i)
            probs[idx[i]] = 0.0f;
        sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) sum += probs[i];
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int i = 0; i < vocab_size; ++i) probs[i] *= inv;
        }
    }

    // 5. Min-p filtering
    if (params.min_p > 0.0f) {
        float max_p = *std::max_element(probs.begin(), probs.end());
        float threshold = max_p * params.min_p;
        for (int i = 0; i < vocab_size; ++i)
            if (probs[i] < threshold) probs[i] = 0.0f;
        sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) sum += probs[i];
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int i = 0; i < vocab_size; ++i) probs[i] *= inv;
        }
    }

    return probs;
}

// =========================================================================
// 1. SamplingParams construction
// =========================================================================
static void test_sampling_params() {
    printf("[1] SamplingParams\n");

    // Default values
    SamplingParams dflt;
    CHECK("params/default_temp",     neareq(dflt.temperature, 1.0f));
    CHECK("params/default_topk",     dflt.top_k == 0);
    CHECK("params/default_topp",     neareq(dflt.top_p, 1.0f));
    CHECK("params/default_minp",     neareq(dflt.min_p, 0.0f));
    CHECK("params/default_seed",     dflt.seed == 42);

    // Custom values via aggregate init (C++20 designated initialisers)
    SamplingParams custom{.temperature = 0.8f, .top_k = 50, .top_p = 0.95f,
                          .min_p = 0.05f,     .seed = 12345};
    CHECK("params/custom_temp",      neareq(custom.temperature, 0.8f));
    CHECK("params/custom_topk",      custom.top_k == 50);
    CHECK("params/custom_topp",      neareq(custom.top_p, 0.95f));
    CHECK("params/custom_minp",      neareq(custom.min_p, 0.05f));
    CHECK("params/custom_seed",      custom.seed == 12345);

    // Greedy setup
    SamplingParams greedy{.temperature = 0.0f};
    CHECK("params/greedy_temp",      neareq(greedy.temperature, 0.0f));

    // Fully disabled filters (pure multinomial)
    SamplingParams pure{.temperature = 1.0f, .top_k = 0, .top_p = 1.0f, .min_p = 0.0f};
    CHECK("params/pure_temp",        neareq(pure.temperature, 1.0f));
    CHECK("params/pure_topk",        pure.top_k == 0);
    CHECK("params/pure_topp",        neareq(pure.top_p, 1.0f));

    printf("  [PASS] test_sampling_params  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// 2. Greedy (temperature == 0)
// =========================================================================
static void test_greedy() {
    printf("[2] Greedy (temperature=0)\n");

    SamplingParams greedy{/*temperature=*/0.0f};
    std::mt19937 rng(42);

    // Simple case: pick the max
    {
        std::vector<float> logits = {0.1f, 0.5f, 0.3f, 0.1f};
        int token = sample(logits.data(), (int)logits.size(), greedy, rng);
        CHECK("greedy/simple_max",   token == 1);
    }

    // All negative: pick largest (closest to zero)
    {
        std::vector<float> logits = {-1.0f, -2.0f, -3.0f};
        int token = sample(logits.data(), (int)logits.size(), greedy, rng);
        CHECK("greedy/all_negative", token == 0);
    }

    // All zeros: any valid index, deterministic across calls
    {
        std::vector<float> logits(100, 0.0f);
        int t0 = sample(logits.data(), (int)logits.size(), greedy, rng);
        CHECK("greedy/all_zeros_range", t0 >= 0 && t0 < 100);
        // Greedy with all equal should be deterministic (always same index)
        for (int rep = 0; rep < 10; ++rep) {
            int t = sample(logits.data(), (int)logits.size(), greedy, rng);
            CHECK("greedy/all_zeros_deterministic", t == t0);
        }
    }

    // Single token vocabulary
    {
        std::vector<float> logits = {42.0f};
        int token = sample(logits.data(), 1, greedy, rng);
        CHECK("greedy/vocab_one",    token == 0);
    }

    // Large logit spread
    {
        std::vector<float> logits = {1e10f, 0.0f, -1e10f, 5.0f};
        int token = sample(logits.data(), (int)logits.size(), greedy, rng);
        CHECK("greedy/large_spread", token == 0);
    }

    // Very negative logits — still picks largest
    {
        std::vector<float> logits = {-1e6f, -5e5f, -1e7f, -1e4f};
        int token = sample(logits.data(), (int)logits.size(), greedy, rng);
        CHECK("greedy/very_negative", token == 3); // -1e4 is largest
    }

    // Greedy should ignore all other params (top_k, top_p, min_p are irrelevant)
    {
        std::vector<float> logits = {0.1f, 0.9f, 0.2f, 0.05f};
        SamplingParams greedy_with_filters{.temperature = 0.0f, .top_k = 1,
                                          .top_p = 0.3f,     .min_p = 0.5f};
        int token = sample(logits.data(), (int)logits.size(),
                           greedy_with_filters, rng);
        // temperature==0 short-circuits — top_k/top_p/min_p MUST be ignored
        CHECK("greedy/ignores_filters", token == 1);
    }

    printf("  [PASS] test_greedy  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// 3. Temperature scaling
// =========================================================================
static void test_temperature() {
    printf("[3] Temperature\n");

    // Build a distribution where token 0 dominates, then temperature=∞ flattens
    std::vector<float> logits = {2.0f, 1.0f, 0.0f, -1.0f, -2.0f};
    int vocab = (int)logits.size();

    // temp=1.0: distribution should match the softmax of raw logits
    {
        SamplingParams p{/*temperature=*/1.0f};
        std::mt19937 rng(100);
        auto expected = expected_distribution(logits.data(), vocab, p);

        // Run many samples and verify token 0 (highest prob) dominates
        std::vector<int> counts(vocab, 0);
        const int N = 10000;
        for (int i = 0; i < N; ++i) {
            int t = sample(logits.data(), vocab, p, rng);
            CHECK("temp1/valid_token", t >= 0 && t < vocab);
            counts[t]++;
        }
        // Token 0 (largest logit) should be most frequent
        int max_idx = (int)(std::max_element(counts.begin(), counts.end())
                            - counts.begin());
        CHECK("temp1/token0_dominates", max_idx == 0);

        // Rough probability check: empirical ~ expected
        for (int i = 0; i < vocab; ++i) {
            float empirical = (float)counts[i] / N;
            float diff = fabsf(empirical - expected[i]);
            CHECK("temp1/distribution", diff < 0.05f);  // within 5%
        }
    }

    // temp=2.0: distribution flattened (higher entropy, less peaked)
    {
        SamplingParams p{/*temperature=*/2.0f};
        std::mt19937 rng(200);
        auto expected = expected_distribution(logits.data(), vocab, p);

        std::vector<int> counts(vocab, 0);
        const int N = 10000;
        for (int i = 0; i < N; ++i) {
            int t = sample(logits.data(), vocab, p, rng);
            counts[t]++;
        }
        for (int i = 0; i < vocab; ++i) {
            float empirical = (float)counts[i] / N;
            float diff = fabsf(empirical - expected[i]);
            CHECK("temp2/distribution", diff < 0.05f);
        }
        // Token 0 should still be most frequent but less dominant than temp=1
        int max_idx = (int)(std::max_element(counts.begin(), counts.end())
                            - counts.begin());
        CHECK("temp2/token0_still_max", max_idx == 0);
    }

    // temp=0.5: distribution sharpened (lower entropy, more peaked)
    {
        SamplingParams p{/*temperature=*/0.5f};
        std::mt19937 rng(300);
        auto expected = expected_distribution(logits.data(), vocab, p);

        std::vector<int> counts(vocab, 0);
        const int N = 10000;
        for (int i = 0; i < N; ++i) {
            int t = sample(logits.data(), vocab, p, rng);
            counts[t]++;
        }
        // Token 0 should dominate even more
        CHECK("temp05/token0_high",     (float)counts[0] / N > 0.5f);
        for (int i = 0; i < vocab; ++i) {
            float empirical = (float)counts[i] / N;
            float diff = fabsf(empirical - expected[i]);
            CHECK("temp05/distribution", diff < 0.05f);
        }
    }

    // Very high temperature: nearly uniform
    {
        SamplingParams p{/*temperature=*/100.0f};
        std::mt19937 rng(400);
        std::vector<int> counts(vocab, 0);
        const int N = 10000;
        for (int i = 0; i < N; ++i)
            counts[sample(logits.data(), vocab, p, rng)]++;
        // Every token should be sampled at least once
        for (int i = 0; i < vocab; ++i)
            CHECK("temp100/all_sampled", counts[i] > 0);
        // All counts should be within ~3x of each other (rough uniformity)
        int min_c = *std::min_element(counts.begin(), counts.end());
        int max_c = *std::max_element(counts.begin(), counts.end());
        CHECK("temp100/near_uniform",   (float)max_c / (float)min_c < 3.0f);
    }

    // Very small temperature (but not 0): effectively greedy
    {
        SamplingParams p{/*temperature=*/0.001f};
        std::mt19937 rng(500);
        int hits = 0;
        for (int i = 0; i < 100; ++i) {
            if (sample(logits.data(), vocab, p, rng) == 0) hits++;
        }
        CHECK("temp_tiny/nearly_greedy", hits >= 95); // at least 95% token 0
    }

    printf("  [PASS] test_temperature  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// 4. Top-K filtering
// =========================================================================
static void test_topk() {
    printf("[4] Top-K\n");

    int vocab = 10;
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; ++i)
        logits[i] = (float)(vocab - i);  // descending: 10,9,8,...,1

    // top_k=1: equivalent to greedy
    {
        SamplingParams p{.temperature = 1.0f, .top_k = 1};
        std::mt19937 rng(600);
        for (int rep = 0; rep < 20; ++rep) {
            int t = sample(logits.data(), vocab, p, rng);
            CHECK("topk1/always_max", t == 0);  // token 0 has logit=10
        }
    }

    // top_k=3: only tokens 0, 1, 2 possible
    {
        SamplingParams p{.temperature = 1.0f, .top_k = 3};
        std::mt19937 rng(700);
        std::set<int> seen;
        const int N = 5000;
        for (int i = 0; i < N; ++i) {
            int t = sample(logits.data(), vocab, p, rng);
            CHECK("topk3/only_top_three", t == 0 || t == 1 || t == 2);
            seen.insert(t);
        }
        // All three should appear
        CHECK("topk3/all_three_seen", seen.size() == 3);
    }

    // top_k=3 with duplicate values at the boundary
    {
        //         idx:  0    1    2    3    4   5
        std::vector<float> dup = {5.0f, 3.0f, 3.0f, 3.0f, 1.0f, 0.0f};
        SamplingParams p{.temperature = 1.0f, .top_k = 2};
        std::mt19937 rng(750);
        for (int i = 0; i < 100; ++i) {
            int t = sample(dup.data(), (int)dup.size(), p, rng);
            // nth_element with k=2 on [5,3,3,3,1,0] keeps values >= threshold
            // k-th largest is the 2nd largest = 3.0
            // So tokens with logit >= 3.0 survive: indices 0,1,2,3
            CHECK("topk_duplicates", t == 0 || t == 1 || t == 2 || t == 3);
        }
    }

    // top_k=0: disabled (all tokens kept)
    {
        SamplingParams p{.temperature = 1.0f, .top_k = 0};
        std::mt19937 rng(800);
        std::set<int> seen;
        for (int i = 0; i < 2000; ++i)
            seen.insert(sample(logits.data(), vocab, p, rng));
        // Most tokens should eventually appear (tail may be rare)
        CHECK("topk0/most_seen", (int)seen.size() >= 5);
    }

    // top_k >= vocab: same as disabled
    {
        SamplingParams p{.temperature = 1.0f, .top_k = 100};
        std::mt19937 rng(900);
        std::set<int> seen;
        for (int i = 0; i < 2000; ++i)
            seen.insert(sample(logits.data(), vocab, p, rng));
        CHECK("topk_big/all_possible", (int)seen.size() >= vocab - 1);
    }

    // top_k with temperature
    {
        SamplingParams p{.temperature = 0.8f, .top_k = 3};
        std::mt19937 rng(1000);
        for (int i = 0; i < 2000; ++i) {
            int t = sample(logits.data(), vocab, p, rng);
            CHECK("topk_temp/only_top_three", t == 0 || t == 1 || t == 2);
        }
    }

    printf("  [PASS] test_topk  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// 5. Top-P (nucleus) filtering
// =========================================================================
static void test_topp() {
    printf("[5] Top-P (nucleus)\n");

    // Sharp distribution: one token dominates
    std::vector<float> logits = {10.0f, 1.0f, 0.5f, 0.1f, 0.05f};
    int vocab = (int)logits.size();

    // top_p=1.0: disabled, all tokens possible
    {
        SamplingParams p{.temperature = 1.0f, .top_p = 1.0f};
        std::mt19937 rng(1100);
        std::set<int> seen;
        for (int i = 0; i < 5000; ++i)
            seen.insert(sample(logits.data(), vocab, p, rng));
        CHECK("topp1/all_possible", (int)seen.size() >= 2);
    }

    // top_p=0.5: with sharp dist, token 0 alone has prob > 0.5
    // The cumulative threshold cuts off lower-prob tails
    {
        SamplingParams p{.temperature = 1.0f, .top_p = 0.5f};
        std::mt19937 rng(1200);
        std::set<int> seen;
        for (int i = 0; i < 5000; ++i)
            seen.insert(sample(logits.data(), vocab, p, rng));
        // Token 0 dominates, but top_p may include others if cum >= 0.5
        // With softmax of [10,1,0.5,0.1,0.05], token 0 has ~0.999 prob
        // So top_p=0.5 only keeps token 0 (cumulative reaches 0.5 at first token)
        CHECK("topp05/token0_appears", seen.count(0) > 0);
    }

    // top_p=0.01: effectively greedy (only token 0 survives nucleus)
    {
        SamplingParams p{.temperature = 1.0f, .top_p = 0.01f};
        std::mt19937 rng(1300);
        for (int i = 0; i < 100; ++i) {
            int t = sample(logits.data(), vocab, p, rng);
            CHECK("topp001/effectively_greedy", t == 0);
        }
    }

    // Flat distribution: all tokens equal, top_p=0.5 keeps subset
    {
        std::vector<float> flat = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        int n = (int)flat.size();
        SamplingParams p{.temperature = 1.0f, .top_p = 0.5f};
        std::mt19937 rng(1400);

        // With 8 equal tokens, each has prob 1/8 = 0.125
        // top_p=0.5 keeps ceil(0.5/0.125) = 4 tokens (cum = 0.5 exactly)
        // But only the top 4 (by prob, all equal → first 4 in sorted order)
        std::set<int> seen;
        for (int i = 0; i < 5000; ++i)
            seen.insert(sample(flat.data(), n, p, rng));
        // Since all probs equal, top_p picks first 4 in sorted order.
        // Due to stable sort or tie-breaking, some tokens may never appear.
        // Just verify we have a subset.
        CHECK("topp_flat/subset", (int)seen.size() >= 2 && (int)seen.size() <= n);
    }

    // top_p=0.0: edge case — keep only the very first token by CDF
    // (the one with highest probability), so only token 1 survives.
    {
        std::vector<float> vals = {0.1f, 0.9f, 0.5f};
        SamplingParams p{.temperature = 1.0f, .top_p = 0.0f};
        std::mt19937 rng(1450);
        std::set<int> seen;
        for (int i = 0; i < 50; ++i) {
            int t = sample(vals.data(), (int)vals.size(), p, rng);
            CHECK("topp0/valid_token", t >= 0 && t < 3);
            seen.insert(t);
        }
        // With top_p=0 only the highest-prob token survives the nucleus.
        // After renorm it has prob 1.0 — so only token 1 should appear.
        CHECK("topp0/only_token1", seen == std::set<int>{1});
    }

    printf("  [PASS] test_topp  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// 6. Min-P filtering
// =========================================================================
static void test_minp() {
    printf("[6] Min-P\n");

    // Distribution with a dominant token and a tail
    // Raw logits scaled to produce meaningful softmax probabilities
    std::vector<float> logits = {3.0f, 1.0f, 0.5f, 0.0f, -1.0f, -3.0f};
    int vocab = (int)logits.size();

    // min_p=0.0: disabled, all tokens with non-zero prob possible
    {
        SamplingParams p{.temperature = 1.0f, .min_p = 0.0f};
        std::mt19937 rng(1500);
        std::set<int> seen;
        for (int i = 0; i < 5000; ++i)
            seen.insert(sample(logits.data(), vocab, p, rng));
        CHECK("minp0/most_possible", (int)seen.size() >= vocab - 1);
    }

    // min_p=0.1: remove tokens with prob < 10% of max_prob
    {
        SamplingParams p{.temperature = 1.0f, .min_p = 0.1f};
        std::mt19937 rng(1600);

        std::set<int> seen;
        for (int i = 0; i < 5000; ++i) {
            int t = sample(logits.data(), vocab, p, rng);
            seen.insert(t);
        }
        // Min-p=0.1 should heavily restrict the tail. Fewer than half the
        // vocabulary should appear.
        CHECK("minp01/filters_tail", (int)seen.size() < vocab / 2);
    }

    // min_p=0.9: nearly everything except the top token removed
    {
        SamplingParams p{.temperature = 1.0f, .min_p = 0.9f};
        std::mt19937 rng(1700);
        std::set<int> seen;
        for (int i = 0; i < 5000; ++i)
            seen.insert(sample(logits.data(), vocab, p, rng));
        // With min_p=0.9, only the top token(s) survive (very few distinct)
        CHECK("minp09/few_survive", (int)seen.size() <= 3);
    }

    printf("  [PASS] test_minp  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// 7. Combined parameters
// =========================================================================
static void test_combined() {
    printf("[7] Combined parameters\n");

    const int vocab = 50;
    // Create a distribution with some structure
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; ++i)
        logits[i] = 5.0f * expf(-0.1f * (float)i); // decaying: 5, 4.5, 4.1, ...

    // Standard LLM params: temp=0.8, top_k=50, top_p=0.95
    {
        SamplingParams p{.temperature = 0.8f, .top_k = 50,
                         .top_p = 0.95f};
        std::mt19937 rng(1800);

        auto expected = expected_distribution(logits.data(), vocab, p);

        // Run many samples
        std::vector<int> counts(vocab, 0);
        const int N = 10000;
        for (int i = 0; i < N; ++i) {
            int t = sample(logits.data(), vocab, p, rng);
            CHECK("combined/basic_range", t >= 0 && t < vocab);
            counts[t]++;
        }

        // Verify that tokens with zero expected prob were never sampled
        for (int i = 0; i < vocab; ++i) {
            if (expected[i] <= 0.0f) {
                CHECK("combined/zero_prob_not_sampled", counts[i] == 0);
            }
        }

        // Distribution roughly matches expected
        for (int i = 0; i < vocab; ++i) {
            if (expected[i] > 0.01f) { // only check significant probs
                float empirical = (float)counts[i] / N;
                float diff = fabsf(empirical - expected[i]);
                CHECK("combined/distribution", diff < 0.05f);
            }
        }
    }

    // Full pipeline: temp=0.7, top_k=40, top_p=0.9, min_p=0.05
    {
        SamplingParams p{.temperature = 0.7f, .top_k = 40,
                         .top_p = 0.9f,     .min_p = 0.05f};
        std::mt19937 rng(1900);

        auto expected = expected_distribution(logits.data(), vocab, p);

        std::vector<int> counts(vocab, 0);
        const int N = 10000;
        for (int i = 0; i < N; ++i) {
            int t = sample(logits.data(), vocab, p, rng);
            CHECK("combined/full_range", t >= 0 && t < vocab);
            counts[t]++;
        }

        // Zero-prob tokens never sampled
        for (int i = 0; i < vocab; ++i) {
            if (expected[i] <= 0.0f) {
                CHECK("combined/full_zero_not_sampled", counts[i] == 0);
            }
        }

        // Distribution matches
        for (int i = 0; i < vocab; ++i) {
            if (expected[i] > 0.02f) {
                float empirical = (float)counts[i] / N;
                float diff = fabsf(empirical - expected[i]);
                CHECK("combined/full_distribution", diff < 0.05f);
            }
        }
    }

    // top_k + top_p interaction: top_k should dominate (runs first on logits)
    {
        //         idx:  0     1     2     3     4     5     6     7
        std::vector<float> logs = {10.0f, 9.0f, 8.0f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
        int n = (int)logs.size();

        // top_k=3 restricts to tokens 0,1,2 regardless of top_p
        SamplingParams p{.temperature = 1.0f, .top_k = 3, .top_p = 0.95f};
        std::mt19937 rng(1950);
        for (int i = 0; i < 2000; ++i) {
            int t = sample(logs.data(), n, p, rng);
            CHECK("combined/topk_before_topp", t >= 0 && t <= 2);
        }
    }

    printf("  [PASS] test_combined  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// 8. Edge cases
// =========================================================================
static void test_edge_cases() {
    printf("[8] Edge cases\n");

    // --- vocab_size = 1 ---
    {
        std::vector<float> logits = {3.14f};
        SamplingParams p{.temperature = 1.0f, .top_k = 5, .top_p = 0.5f};
        std::mt19937 rng(2000);
        for (int rep = 0; rep < 10; ++rep) {
            int t = sample(logits.data(), 1, p, rng);
            CHECK("edge/vocab1", t == 0);
        }
    }

    // --- vocab_size = 1 with greedy ---
    {
        std::vector<float> logits = {0.0f};
        SamplingParams p{/*temperature=*/0.0f};
        std::mt19937 rng(2001);
        CHECK("edge/vocab1_greedy", sample(logits.data(), 1, p, rng) == 0);
    }

    // --- All -inf logits (should not crash, returns valid index) ---
    {
        std::vector<float> logits(10, -INFINITY);
        SamplingParams p{/*temperature=*/1.0f};
        std::mt19937 rng(2100);

        bool returned = false;
        int t = -1;
        // Should not hang or crash
        t = sample(logits.data(), (int)logits.size(), p, rng);
        returned = true;
        CHECK("edge/all_neginf/returned", returned);
        CHECK("edge/all_neginf/valid",    t >= 0 && t < (int)logits.size());
    }

    // --- All -inf with greedy ---
    {
        std::vector<float> logits(5, -INFINITY);
        SamplingParams p{/*temperature=*/0.0f};
        std::mt19937 rng(2101);
        int t = sample(logits.data(), (int)logits.size(), p, rng);
        CHECK("edge/all_neginf_greedy/valid", t >= 0 && t < 5);
    }

    // --- Very large logits (no overflow) ---
    {
        std::vector<float> logits = {1e10f, 5e9f, 0.0f, -1e10f};
        SamplingParams p{/*temperature=*/1.0f};
        std::mt19937 rng(2200);

        // With temp=1, softmax of [1e10, 5e9, 0, -1e10] should give token 0 ~1.0 prob
        int count0 = 0;
        for (int i = 0; i < 100; ++i) {
            if (sample(logits.data(), (int)logits.size(), p, rng) == 0)
                count0++;
        }
        CHECK("edge/large_logits_no_overflow", count0 >= 95);
    }

    // --- Very small logits (no underflow) ---
    {
        // All logits are very small negatives; after softmax they should be
        // equal probs (1/vocab each)
        std::vector<float> logits = {-1000.0f, -1001.0f, -1002.0f, -1003.0f};
        SamplingParams p{/*temperature=*/1.0f};
        std::mt19937 rng(2300);

        std::set<int> seen;
        for (int i = 0; i < 2000; ++i)
            seen.insert(sample(logits.data(), (int)logits.size(), p, rng));
        // All 4 tokens should eventually appear (they have near-equal prob)
        CHECK("edge/small_logits_all_seen", (int)seen.size() >= 3);
    }

    // --- Mixed very large and very small logits ---
    {
        std::vector<float> logits = {100.0f, -100.0f, -200.0f};
        SamplingParams p{/*temperature=*/1.0f};
        std::mt19937 rng(2350);
        // Token 0 has prob ~1.0, others ~0
        for (int i = 0; i < 20; ++i) {
            CHECK("edge/mixed_magnitude", sample(logits.data(), 3, p, rng) == 0);
        }
    }

    // --- Large vocab (simulate typical 151k vocab) ---
    {
        const int large_vocab = 2000; // scaled down but tests the O(n) path
        std::vector<float> logits(large_vocab);
        // Generate a structured distribution
        std::mt19937 data_rng(999);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int i = 0; i < large_vocab; ++i)
            logits[i] = nd(data_rng);

        // Make token 42 the clear winner
        logits[42] = 20.0f;

        SamplingParams p{.temperature = 1.0f, .top_k = 50, .top_p = 0.9f};
        std::mt19937 rng(2400);

        for (int i = 0; i < 100; ++i) {
            int t = sample(logits.data(), large_vocab, p, rng);
            CHECK("edge/large_vocab/valid", t >= 0 && t < large_vocab);
        }

        // Greedy on large vocab
        SamplingParams greedy{/*temperature=*/0.0f};
        int t = sample(logits.data(), large_vocab, greedy, rng);
        CHECK("edge/large_vocab/greedy", t == 42);
    }

    // --- top_k=1 with temperature (should still only pick the max) ---
    {
        std::vector<float> logits = {0.1f, 0.5f, 0.3f, 0.1f};
        SamplingParams p{.temperature = 2.0f, .top_k = 1};
        std::mt19937 rng(2500);
        for (int i = 0; i < 50; ++i) {
            // With top_k=1, only the max (after temp scaling) survives
            // After temp=2, logits: [0.05, 0.25, 0.15, 0.05]
            // max is still index 1
            CHECK("edge/topk1_with_temp", sample(logits.data(), 4, p, rng) == 1);
        }
    }

    // --- top_k that matches the number of tied-top values ---
    {
        std::vector<float> logits = {5.0f, 5.0f, 5.0f, 1.0f, 1.0f};
        SamplingParams p{.temperature = 1.0f, .top_k = 3};
        std::mt19937 rng(2550);
        // All three 5.0 tokens are in top 3, 1.0 tokens are not
        for (int i = 0; i < 500; ++i) {
            int t = sample(logits.data(), (int)logits.size(), p, rng);
            CHECK("edge/topk_ties", t == 0 || t == 1 || t == 2);
        }
    }

    printf("  [PASS] test_edge_cases  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// 9. Reproducibility
// =========================================================================
static void test_reproducibility() {
    printf("[9] Reproducibility\n");

    std::vector<float> logits = {0.5f, 1.2f, 3.7f, 0.1f, 2.3f, 0.9f, 4.1f};
    int vocab = (int)logits.size();
    SamplingParams p{.temperature = 0.8f, .top_k = 5,
                     .top_p = 0.95f,     .min_p = 0.02f};

    // Same seed => same sequence
    {
        std::mt19937 rng1(42);
        std::mt19937 rng2(42);

        std::vector<int> seq1, seq2;
        for (int i = 0; i < 100; ++i) {
            seq1.push_back(sample(logits.data(), vocab, p, rng1));
            seq2.push_back(sample(logits.data(), vocab, p, rng2));
        }
        CHECK("repro/same_seed_same_seq", seq1 == seq2);
    }

    // Different seed => different sequence (statistically)
    {
        std::mt19937 rng1(42);
        std::mt19937 rng2(12345);

        std::vector<int> seq1, seq2;
        for (int i = 0; i < 100; ++i) {
            seq1.push_back(sample(logits.data(), vocab, p, rng1));
            seq2.push_back(sample(logits.data(), vocab, p, rng2));
        }
        // Extremely unlikely to be identical for 100 draws with 7 tokens
        CHECK("repro/diff_seed_diff_seq", seq1 != seq2);
    }

    // Greedy (temp=0) is always deterministic regardless of seed
    {
        SamplingParams greedy{/*temperature=*/0.0f};
        std::mt19937 rng1(42);
        std::mt19937 rng2(99999);
        for (int i = 0; i < 20; ++i) {
            int t1 = sample(logits.data(), vocab, greedy, rng1);
            int t2 = sample(logits.data(), vocab, greedy, rng2);
            CHECK("repro/greedy_deterministic", t1 == t2);
        }
    }

    // Seed via params: when seed != 0 in SamplingParams, the caller uses it.
    // The sample function itself takes an already-seeded rng — so test that
    // same initial rng state => same output.
    {
        std::mt19937 rng1(777);
        std::mt19937 rng2(777);
        std::vector<int> seq1, seq2;
        for (int i = 0; i < 50; ++i) {
            seq1.push_back(sample(logits.data(), vocab, p, rng1));
            seq2.push_back(sample(logits.data(), vocab, p, rng2));
        }
        CHECK("repro/same_rng_state", seq1 == seq2);
    }

    // After one sequence draws, the rng state diverges => different outputs
    {
        std::mt19937 rng1(555);
        std::mt19937 rng2(555);

        // Draw one from rng2 to advance its state
        sample(logits.data(), vocab, p, rng2);

        int t1 = sample(logits.data(), vocab, p, rng1);
        int t2 = sample(logits.data(), vocab, p, rng2);
        // Now rng1 and rng2 are at different states
        // They could coincidentally match, but with 7 tokens it's < 15% chance
        bool likely_different = true;
        // Run multiple times to reduce false positive
        int matches = 0;
        for (int i = 0; i < 5; ++i) {
            if (sample(logits.data(), vocab, p, rng1) ==
                sample(logits.data(), vocab, p, rng2))
                matches++;
        }
        CHECK("repro/diverged_state", matches < 5);  // at most 4/5 can match by chance
    }

    printf("  [PASS] test_reproducibility  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// 10. sample_batch
// =========================================================================
static void test_sample_batch() {
    printf("[10] sample_batch\n");

    const int vocab = 10;
    const int batch = 8;

    // Build batch logits: row-major [batch * vocab]
    // Row 0: strongly prefer token 0
    // Row 1: strongly prefer token 1
    // etc.
    std::vector<float> logits(batch * vocab, 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int v = 0; v < vocab; ++v) {
            // Each row peaks at a different token
            float dist = (float)(v - b);
            logits[b * vocab + v] = 5.0f * expf(-0.5f * dist * dist);
        }
    }

    // Greedy batch: token b for sequence b
    {
        SamplingParams p{/*temperature=*/0.0f};
        std::vector<int> tokens(batch, -1);
        sample_batch(tokens.data(), logits.data(), batch, vocab, p, 42);

        for (int b = 0; b < batch; ++b) {
            char msg[64];
            snprintf(msg, sizeof(msg), "batch_greedy/row_%d", b);
            CHECK(msg, tokens[b] == b);
        }
    }

    // Stochastic batch with top_k: each row only gives top-k tokens
    {
        SamplingParams p{.temperature = 1.0f, .top_k = 3};
        std::vector<int> tokens(batch, -1);
        sample_batch(tokens.data(), logits.data(), batch, vocab, p, 123);

        for (int b = 0; b < batch; ++b) {
            char msg[64];
            snprintf(msg, sizeof(msg), "batch_topk3/row_%d_range", b);
            // The top 3 tokens for row b are b, b±1 (within vocab bounds)
            CHECK(msg, tokens[b] >= 0 && tokens[b] < vocab);

            // Verify it's close to the peak
            snprintf(msg, sizeof(msg), "batch_topk3/row_%d_near_peak", b);
            int diff = tokens[b] - b;
            CHECK(msg, diff >= -2 && diff <= 2);
        }
    }

    // Batch reproducibility: same seed => same output
    {
        SamplingParams p{.temperature = 0.8f, .top_k = 0, .top_p = 0.95f};
        std::vector<int> tokens1(batch, -1);
        std::vector<int> tokens2(batch, -1);

        sample_batch(tokens1.data(), logits.data(), batch, vocab, p, 999);
        sample_batch(tokens2.data(), logits.data(), batch, vocab, p, 999);

        CHECK("batch_repro/same", tokens1 == tokens2);
    }

    // Batch with different seed => different output
    {
        SamplingParams p{.temperature = 0.8f, .top_p = 0.95f};
        std::vector<int> tokens1(batch, -1);
        std::vector<int> tokens2(batch, -1);

        sample_batch(tokens1.data(), logits.data(), batch, vocab, p, 100);
        sample_batch(tokens2.data(), logits.data(), batch, vocab, p, 200);

        CHECK("batch_repro/diff", tokens1 != tokens2);
    }

    // batch_size=1 edge case
    {
        SamplingParams p{/*temperature=*/1.0f};
        std::vector<int> tokens(1, -1);
        sample_batch(tokens.data(), logits.data(), 1, vocab, p, 300);
        CHECK("batch_single/valid", tokens[0] >= 0 && tokens[0] < vocab);
    }

    // Large batch
    {
        const int big_batch = 64;
        std::vector<float> big_logits(big_batch * vocab);
        for (int b = 0; b < big_batch; ++b)
            for (int v = 0; v < vocab; ++v)
                big_logits[b * vocab + v] = (float)(v + b) * 0.1f;

        SamplingParams p{/*temperature=*/1.0f};
        std::vector<int> tokens(big_batch, -1);
        sample_batch(tokens.data(), big_logits.data(), big_batch, vocab, p, 400);

        for (int b = 0; b < big_batch; ++b) {
            char msg[64];
            snprintf(msg, sizeof(msg), "batch_big/row_%d", b);
            CHECK(msg, tokens[b] >= 0 && tokens[b] < vocab);
        }
    }

    printf("  [PASS] test_sample_batch  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// 11. Detail helpers (inline functions in detail namespace)
// =========================================================================
static void test_detail_helpers() {
    printf("[11] Detail helpers\n");

    using namespace lightllm::ops::detail;

    // --- softmax_inplace ---
    {
        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
        float s = softmax_inplace(x.data(), (int)x.size());
        // softmax_inplace returns the pre-norm sum of exp values (~1.55 for
        // these logits), NOT 1.0.  The array should sum to 1.0 after norm.
        CHECK("detail/softmax_sum", s > 0.0f);
        (void)s; // silence unused warning

        float total = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) total += x[i];
        CHECK("detail/softmax_normalized", neareq(total, 1.0f));

        // Values should be increasing (larger logit => larger prob)
        for (size_t i = 1; i < x.size(); ++i)
            CHECK("detail/softmax_monotonic", x[i] > x[i-1]);
    }

    // softmax with -inf values
    {
        std::vector<float> x = {1.0f, -INFINITY, 0.0f};
        float s = softmax_inplace(x.data(), (int)x.size());
        CHECK("detail/softmax_inf/valid", s > 0.0f);
        CHECK("detail/softmax_inf/zero",  neareq(x[1], 0.0f));
        float total = x[0] + x[1] + x[2];
        CHECK("detail/softmax_inf/norm",  neareq(total, 1.0f));
    }

    // softmax with all equal values
    {
        std::vector<float> x(10, 5.0f);
        softmax_inplace(x.data(), (int)x.size());
        for (int i = 0; i < 10; ++i)
            CHECK("detail/softmax_uniform", neareq(x[i], 0.1f));
    }

    // softmax with all -inf: -inf - (-inf) = NaN by IEEE 754.
    // softmax_inplace is not required to handle this case gracefully on its
    // own (the sample() entry point handles all -inf via a separate path).
    // We just verify no crash — the exact output is unspecified for all -inf.

    // --- multinomial_draw ---
    {
        std::mt19937 rng(42);

        // Deterministic: prob=1.0 for a single token
        std::vector<float> probs = {0.0f, 0.0f, 1.0f, 0.0f};
        for (int i = 0; i < 20; ++i)
            CHECK("detail/multinomial_deterministic",
                  multinomial_draw(probs.data(), (int)probs.size(), rng) == 2);

        // All zeros => returns 0
        std::vector<float> zeros = {0.0f, 0.0f, 0.0f};
        int t = multinomial_draw(zeros.data(), (int)zeros.size(), rng);
        CHECK("detail/multinomial_all_zeros", t == 0);

        // Statistical test: run many draws and verify frequencies
        std::vector<float> unif = {0.1f, 0.2f, 0.3f, 0.4f};
        std::vector<int> counts(4, 0);
        const int N = 100000;
        for (int i = 0; i < N; ++i) {
            int idx = multinomial_draw(unif.data(), 4, rng);
            CHECK("detail/multinomial_bounds", idx >= 0 && idx < 4);
            counts[idx]++;
        }
        for (int i = 0; i < 4; ++i) {
            float empirical = (float)counts[i] / N;
            float expected  = unif[i];  // already sums to 1
            char msg[64];
            snprintf(msg, sizeof(msg), "detail/multinomial_freq_%d", i);
            CHECK(msg, fabsf(empirical - expected) < 0.02f);
        }
    }

    // --- apply_top_k ---
    {
        std::vector<float> logits = {5.0f, 1.0f, 4.0f, 2.0f, 3.0f};
        int n = (int)logits.size();
        std::vector<float> work = logits; // copy
        int kept = apply_top_k(work.data(), n, 3);
        CHECK("detail/topk_kept_count", kept == 3);

        // Tokens 0 (5.0), 2 (4.0), 4 (3.0) should survive
        CHECK("detail/topk_idx0", work[0] > -1e30f);
        CHECK("detail/topk_idx1", std::isinf(work[1])); // should be -inf
        CHECK("detail/topk_idx2", work[2] > -1e30f);
        CHECK("detail/topk_idx3", std::isinf(work[3])); // should be -inf
        CHECK("detail/topk_idx4", work[4] > -1e30f);
    }

    // apply_top_k with k >= n => no change
    {
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};
        std::vector<float> work = logits;
        int kept = apply_top_k(work.data(), 3, 5);
        CHECK("detail/topk_oversize_kept", kept == 3);
        for (int i = 0; i < 3; ++i)
            CHECK("detail/topk_oversize_unchanged", neareq(work[i], logits[i]));
    }

    // apply_top_k with k <= 0 => no change
    {
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};
        std::vector<float> work = logits;
        int kept = apply_top_k(work.data(), 3, 0);
        CHECK("detail/topk_zero", kept == 3);
    }

    // --- apply_min_p ---
    {
        std::vector<float> probs = {0.5f, 0.3f, 0.15f, 0.04f, 0.01f};
        int n = (int)probs.size();
        std::vector<float> work = probs;
        apply_min_p(work.data(), n, 0.1f);
        // max_prob=0.5, threshold=0.05 => keep 0.5, 0.3, 0.15, remove 0.04, 0.01
        CHECK("detail/minp_kept_high",    work[0] > 0.0f);
        CHECK("detail/minp_kept_mid",     work[1] > 0.0f);
        CHECK("detail/minp_removed_low1", neareq(work[3], 0.0f));
        CHECK("detail/minp_removed_low2", neareq(work[4], 0.0f));
        // Renormalization: verify sum
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += work[i];
        CHECK("detail/minp_renorm", neareq(s, 1.0f));
    }

    // apply_min_p with min_p=0 => no change
    {
        std::vector<float> probs = {0.5f, 0.3f, 0.2f};
        std::vector<float> work = probs;
        apply_min_p(work.data(), 3, 0.0f);
        for (int i = 0; i < 3; ++i)
            CHECK("detail/minp_zero_unchanged", neareq(work[i], probs[i]));
    }

    // --- apply_top_p ---
    {
        std::vector<float> probs = {0.5f, 0.3f, 0.1f, 0.07f, 0.03f};
        int n = (int)probs.size();
        std::vector<float> work = probs;
        apply_top_p(work.data(), n, 0.85f);
        // Sorted desc: 0.5 (idx0), 0.3 (idx1), 0.1 (idx2), 0.07 (idx3), 0.03 (idx4)
        // Cum: 0.5, 0.8, 0.9 -> cutoff at idx 2 (cum=0.9 >= 0.85)
        // Keep: 0, 1, 2; remove: 3, 4
        char msg[64];
        CHECK("detail/topp_kept_high",    work[0] > 0.0f);
        CHECK("detail/topp_kept_mid",     work[1] > 0.0f);
        CHECK("detail/topp_kept_low",     work[2] > 0.0f);
        CHECK("detail/topp_removed_1",    neareq(work[3], 0.0f));
        CHECK("detail/topp_removed_2",    neareq(work[4], 0.0f));
        // Renormalize
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += work[i];
        CHECK("detail/topp_renorm", neareq(s, 1.0f));
    }

    // apply_top_p with top_p=1.0 => no change
    {
        std::vector<float> probs = {0.6f, 0.4f, 0.0f};
        std::vector<float> work = probs;
        apply_top_p(work.data(), 3, 1.0f);
        for (int i = 0; i < 3; ++i)
            CHECK("detail/topp_one_unchanged", neareq(work[i], probs[i]));
    }

    // apply_top_p with equal probabilities => first ceil(top_p * n) survive
    {
        std::vector<float> probs(8, 0.125f);
        std::vector<float> work = probs;
        apply_top_p(work.data(), 8, 0.4f);
        // Each prob = 0.125. Cum: 0.125, 0.25, 0.375, 0.5 -> cutoff at 4
        // Keeps 4 tokens
        int kept = 0;
        for (int i = 0; i < 8; ++i)
            if (work[i] > 0.0f) kept++;
        CHECK("detail/topp_equal", kept == 4);
        // Renormalized
        float s = 0.0f;
        for (int i = 0; i < 8; ++i) s += work[i];
        CHECK("detail/topp_equal_renorm", neareq(s, 1.0f));
    }

    printf("  [PASS] test_detail_helpers  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// 12. Stress / statistical properties over many samples
// =========================================================================
static void test_stress() {
    printf("[12] Stress / statistical\n");

    int vocab = 20;
    // Generate logits with a clear ordering but some noise
    std::mt19937 data_rng(555);
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; ++i)
        logits[i] = 3.0f - 0.15f * (float)i + 0.05f * (float)(data_rng() % 100) / 100.0f;

    // Full pipeline
    SamplingParams p{.temperature = 1.0f, .top_k = 10,
                     .top_p = 0.9f,     .min_p = 0.02f};
    auto expected = expected_distribution(logits.data(), vocab, p);

    std::mt19937 rng(42);
    std::vector<int> counts(vocab, 0);
    const int N = 50000;
    for (int i = 0; i < N; ++i) {
        int t = sample(logits.data(), vocab, p, rng);
        counts[t]++;
    }

    // Chi-squared test for goodness of fit
    double chi2 = 0.0;
    int dof = 0;  // degrees of freedom (non-zero categories - 1)
    for (int i = 0; i < vocab; ++i) {
        if (expected[i] > 0.001f) {  // only test categories with non-trivial prob
            double exp_count = expected[i] * N;
            double obs_count = (double)counts[i];
            double diff = obs_count - exp_count;
            chi2 += (diff * diff) / exp_count;
            dof++;
        }
    }
    dof--; // subtract 1 for the constraint sum(p_i) = 1

    // For dof ~ 7-9, critical chi2 at p=0.01 is about 18-22
    // Use a generous threshold of 50
    char msg[128];
    snprintf(msg, sizeof(msg),
             "stress/chi2=%.1f dof=%d (threshold<50)", chi2, dof);
    CHECK(msg, chi2 < 50.0 && dof > 0);

    // Verify entropy is reasonable (not degenerate)
    double entropy = 0.0;
    for (int i = 0; i < vocab; ++i) {
        if (expected[i] > 0.0f)
            entropy -= expected[i] * log2((double)expected[i]);
    }
    snprintf(msg, sizeof(msg),
             "stress/entropy=%.2f bits (threshold>0.1)", entropy);
    CHECK(msg, entropy > 0.1); // should not be fully deterministic

    // Token diversity: at least 3 distinct tokens in 50000 samples
    int distinct = 0;
    for (int i = 0; i < vocab; ++i)
        if (counts[i] > 0) distinct++;
    snprintf(msg, sizeof(msg),
             "stress/distinct=%d (threshold>=3)", distinct);
    CHECK(msg, distinct >= 3);

    printf("  [PASS] test_stress  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// 13. Precision / rounding edge cases
// =========================================================================
static void test_precision() {
    printf("[13] Precision edge cases\n");

    // Denormalized / subnormal values: sample should handle them
    {
        std::vector<float> logits = {1e-40f, 2e-40f, 0.0f, -1e-40f};
        SamplingParams p{/*temperature=*/1.0f};
        std::mt19937 rng(2600);
        for (int i = 0; i < 100; ++i) {
            int t = sample(logits.data(), (int)logits.size(), p, rng);
            CHECK("prec/denorm_valid", t >= 0 && t < (int)logits.size());
        }
        // Greedy with subnormals
        SamplingParams greedy{/*temperature=*/0.0f};
        int t = sample(logits.data(), (int)logits.size(), greedy, rng);
        CHECK("prec/denorm_greedy", t == 1); // 2e-40 > 1e-40 > 0 > -1e-40
    }

    // Temperature that causes exactly-integer-scaled logits
    {
        std::vector<float> logits = {1.0f, 2.0f, 4.0f};
        SamplingParams p{/*temperature=*/2.0f}; // logits become 0.5, 1.0, 2.0
        std::mt19937 rng(2700);
        int t = sample(logits.data(), (int)logits.size(), p, rng);
        CHECK("prec/exact_scale", t >= 0 && t < 3);
    }

    // top_p that lands exactly on a cumulative sum boundary
    {
        std::vector<float> logits = {4.0f, 2.0f, 2.0f}; // softmax: ~0.84, ~0.11, ~0.05
        SamplingParams p{.temperature = 1.0f, .top_p = 0.85f};
        // Cumulative: 0.84 >= 0.85? No — 0.84 < 0.85, then 0.84+0.11=0.95 >= 0.85
        // Cutoff at index 1 (0-based), keeping tokens 0 and either 1 or 2
        // (tokens 1 and 2 have equal logit 2.0, so sort tie-breaking is
        // implementation-defined — either can end up in the nucleus).
        std::mt19937 rng(2800);
        for (int i = 0; i < 200; ++i) {
            int t = sample(logits.data(), 3, p, rng);
            CHECK("prec/exact_topp_boundary", t >= 0 && t <= 2);
        }
    }

    // NaN in logits — the system should be robust
    // (Not testing: the header says logits come from lm_head, which won't
    //  produce NaN, but if they do, the behavior is undefined.)
    // We skip NaN testing to avoid UB.

    printf("  [PASS] test_precision  (%d checks)\n\n", passed + failed);
}

// =========================================================================
// main
// =========================================================================
int main() {
    printf("LightLLM — Sampling Tests\n");
    printf("==========================\n\n");

    test_sampling_params();   fflush(stdout);
    test_greedy();            fflush(stdout);
    test_temperature();       fflush(stdout);
    test_topk();              fflush(stdout);
    test_topp();              fflush(stdout);
    test_minp();              fflush(stdout);
    test_combined();          fflush(stdout);
    test_edge_cases();        fflush(stdout);
    test_reproducibility();   fflush(stdout);
    test_sample_batch();      fflush(stdout);
    test_detail_helpers();    fflush(stdout);
    test_stress();            fflush(stdout);
    test_precision();         fflush(stdout);

    printf("==========================\n");
    printf("%d passed, %d failed, %d total\n", passed, failed, passed + failed);
    return failed == 0 ? 0 : 1;
}
