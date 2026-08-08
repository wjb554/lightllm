/// GPU lm_head correctness tests — verifies the lm_head_logits kernel against
/// CPU reference, exercises edge cases, validates greedy consistency, and
/// measures performance.
///
/// See include/lightllm/ops/lm_head.h for the kernel architecture docs.

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>

#include "lightllm/tensor.h"
#include "lightllm/ops/lm_head.h"

using namespace lightllm;
using namespace lightllm::ops;

// ---------------------------------------------------------------------------
// Test framework
// ---------------------------------------------------------------------------
static int passed = 0, failed = 0;

static void CHECK(const char* name, bool cond) {
    if (cond) {
        passed++;
    } else {
        failed++;
        fprintf(stderr, "  FAIL: %s\n", name);
    }
}

static bool approx(float a, float b, float tol = 1e-3f) {
    return fabsf(a - b) < tol;
}

template <typename T>
static std::vector<T> to_cpu(const Tensor& t) {
    std::vector<T> v(t.numel());
    t.copy_to(v.data(), t.nbytes());
    return v;
}

// ---------------------------------------------------------------------------
// CPU reference: double-loop dot products (same algorithm as the kernel but
// host-side, no parallelism, for ground-truth comparison).
// ---------------------------------------------------------------------------
static std::vector<float> cpu_lm_head(const float* hidden, int D,
                                      const float* embed, int vocab) {
    std::vector<float> logits(vocab);
    for (int v = 0; v < vocab; ++v) {
        const float* row = embed + (size_t)v * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d)
            dot += hidden[d] * row[d];
        logits[v] = dot;
    }
    return logits;
}

// ---------------------------------------------------------------------------
// Convenience: create random float tensor on GPU from host vector.
// ---------------------------------------------------------------------------
static Tensor make_gpu_tensor(const std::vector<int>& shape,
                              const std::vector<float>& data) {
    Tensor t(shape, DType::F32, Device::CUDA);
    t.copy_from(data.data(), data.size() * sizeof(float));
    return t;
}

// ---------------------------------------------------------------------------
// Convenience: fill host vector with uniform random floats in [lo, hi).
// ---------------------------------------------------------------------------
static void rand_fill(std::vector<float>& v, float lo, float hi) {
    for (size_t i = 0; i < v.size(); ++i) {
        float r = (float)rand() / (float)RAND_MAX;
        v[i] = lo + r * (hi - lo);
    }
}

// ===========================================================================
// SECTION 1: Small correctness  (D=16, vocab=32)
// ===========================================================================
static void test_small_correctness() {
    printf("=== SECTION 1: Small Correctness (D=16, V=32) ===\n");

    const int D = 16, V = 32;

    std::vector<float> h_hidden(D);
    std::vector<float> h_embed(V * D);

    rand_fill(h_hidden, -1.0f, 1.0f);
    rand_fill(h_embed, -1.0f, 1.0f);

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto gpu_logits = lm_head_logits(hidden, embed);
    auto h_gpu = to_cpu<float>(gpu_logits);
    auto h_cpu = cpu_lm_head(h_hidden.data(), D, h_embed.data(), V);

    // Verify every entry within 1e-4 absolute tolerance
    int bad = 0;
    float max_err = 0.0f;
    for (int v = 0; v < V; ++v) {
        float err = fabsf(h_gpu[v] - h_cpu[v]);
        if (err > max_err) max_err = err;
        if (err >= 1e-4f) bad++;
    }

    CHECK("small:all_within_tol", bad == 0);
    printf("  max absolute error: %e  (%s)\n", max_err, bad == 0 ? "OK" : "FAIL");
    printf("  [%s] test_small_correctness\n", bad == 0 ? "PASS" : "FAIL");
}

// ===========================================================================
// SECTION 2: Medium correctness  (D=128, vocab=1024)
// ===========================================================================
static void test_medium_correctness() {
    printf("\n=== SECTION 2: Medium Correctness (D=128, V=1024) ===\n");

    const int D = 128, V = 1024;

    std::vector<float> h_hidden(D);
    std::vector<float> h_embed(V * D);

    rand_fill(h_hidden, -1.0f, 1.0f);
    rand_fill(h_embed, -1.0f, 1.0f);

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto gpu_logits = lm_head_logits(hidden, embed);
    auto h_gpu = to_cpu<float>(gpu_logits);
    auto h_cpu = cpu_lm_head(h_hidden.data(), D, h_embed.data(), V);

    // Verify max relative error < 2e-4 (FP32 accumulation over D=128
    // introduces ~1e-5 relative error; values near zero inflate this.)
    int bad = 0;
    float max_rel = 0.0f;
    for (int v = 0; v < V; ++v) {
        float cpu = h_cpu[v];
        float gpu = h_gpu[v];
        float denom = fmaxf(fabsf(cpu), 1e-8f);
        float rel = fabsf(gpu - cpu) / denom;
        if (rel > max_rel) max_rel = rel;
        if (rel >= 2e-4f) bad++;
    }

    CHECK("medium:all_relative_within_tol", bad == 0);
    printf("  max relative error: %e  (%s)\n", max_rel, bad == 0 ? "OK" : "FAIL");
    printf("  first 5: cpu=%.4f gpu=%.4f | cpu=%.4f gpu=%.4f | cpu=%.4f gpu=%.4f | cpu=%.4f gpu=%.4f | cpu=%.4f gpu=%.4f\n",
           h_cpu[0], h_gpu[0], h_cpu[1], h_gpu[1], h_cpu[2], h_gpu[2], h_cpu[3], h_gpu[3], h_cpu[4], h_gpu[4]);
    printf("  [%s] test_medium_correctness\n", bad == 0 ? "PASS" : "FAIL");
}

// ===========================================================================
// SECTION 3: Full-scale correctness  (D=896, vocab=65536)
// ===========================================================================
static void test_full_scale() {
    printf("\n=== SECTION 3: Full-Scale Correctness (D=896, V=65536) ===\n");

    const int D = 896, V = 65536;

    std::vector<float> h_hidden(D);
    std::vector<float> h_embed(V * D);

    rand_fill(h_hidden, -1.0f, 1.0f);
    rand_fill(h_embed, -1.0f, 1.0f);

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto t0 = std::chrono::steady_clock::now();
    auto gpu_logits = lm_head_logits(hidden, embed);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    auto gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    auto h_gpu = to_cpu<float>(gpu_logits);

    // CPU reference only for sampled positions (full 65536 would be slow)
    // Sample: first 100, last 100, middle 100
    auto h_cpu = cpu_lm_head(h_hidden.data(), D, h_embed.data(), V);

    // Check sampled positions
    int bad = 0;
    float max_err = 0.0f;

    auto check_range = [&](int start, int count, const char* label) {
        for (int v = start; v < start + count; ++v) {
            float err = fabsf(h_gpu[v] - h_cpu[v]);
            if (err > max_err) max_err = err;
            if (err >= 0.1f) bad++;
        }
    };

    check_range(0, 100, "first");
    check_range(V / 2 - 50, 100, "middle");
    check_range(V - 100, 100, "last");

    CHECK("fullscale:300_sampled_within_tol", bad == 0);
    printf("  max absolute error (over 300 sampled): %e  (%s)\n", max_err,
           bad == 0 ? "OK" : "FAIL");
    printf("  GPU wall time: %lld us (%.2f ms)\n", gpu_us, gpu_us / 1000.0);

    // Expected: ~544 MB read at ~700 GB/s => ~0.78 ms
    double expected_ms = 0.78;
    printf("  expected: ~%.2f ms (memory-bound, 544 MB / 700 GB/s)\n", expected_ms);

    printf("  [%s] test_full_scale\n", bad == 0 ? "PASS" : "FAIL");
}

// ===========================================================================
// SECTION 4: Edge cases
// ===========================================================================

static void test_edge_d1_v5() {
    printf("\n--- Edge: D=1, vocab=5 ---\n");
    const int D = 1, V = 5;

    std::vector<float> h_hidden = {3.0f};
    std::vector<float> h_embed  = {2.0f, -1.5f, 0.0f, 4.0f, -3.0f};

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto h_gpu = to_cpu<float>(lm_head_logits(hidden, embed));
    auto h_cpu = cpu_lm_head(h_hidden.data(), D, h_embed.data(), V);

    // For D=1, result should be exact (single multiply)
    for (int v = 0; v < V; ++v) {
        CHECK("d1_v5:exact", h_gpu[v] == h_cpu[v]);
    }
    printf("  [PASS] test_edge_d1_v5\n");
}

static void test_edge_v1_d8() {
    printf("\n--- Edge: vocab=1, D=8 ---\n");
    const int D = 8, V = 1;

    std::vector<float> h_hidden(D);
    std::vector<float> h_embed(V * D);

    rand_fill(h_hidden, -1.0f, 1.0f);
    rand_fill(h_embed, -1.0f, 1.0f);

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto h_gpu = to_cpu<float>(lm_head_logits(hidden, embed));
    auto h_cpu = cpu_lm_head(h_hidden.data(), D, h_embed.data(), V);

    CHECK("v1_d8:approx", approx(h_gpu[0], h_cpu[0], 1e-5f));
    printf("  [PASS] test_edge_v1_d8\n");
}

static void test_edge_d7() {
    printf("\n--- Edge: D=7 (not divisible by 4) ---\n");
    const int D = 7, V = 16;

    std::vector<float> h_hidden(D);
    std::vector<float> h_embed(V * D);

    rand_fill(h_hidden, -1.0f, 1.0f);
    rand_fill(h_embed, -1.0f, 1.0f);

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto h_gpu = to_cpu<float>(lm_head_logits(hidden, embed));
    auto h_cpu = cpu_lm_head(h_hidden.data(), D, h_embed.data(), V);

    for (int v = 0; v < V; ++v)
        CHECK("d7:approx", approx(h_gpu[v], h_cpu[v], 1e-4f));
    printf("  [PASS] test_edge_d7\n");
}

static void test_edge_d15() {
    printf("\n--- Edge: D=15 (odd) ---\n");
    const int D = 15, V = 12;

    std::vector<float> h_hidden(D);
    std::vector<float> h_embed(V * D);

    rand_fill(h_hidden, -1.0f, 1.0f);
    rand_fill(h_embed, -1.0f, 1.0f);

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto h_gpu = to_cpu<float>(lm_head_logits(hidden, embed));
    auto h_cpu = cpu_lm_head(h_hidden.data(), D, h_embed.data(), V);

    for (int v = 0; v < V; ++v)
        CHECK("d15:approx", approx(h_gpu[v], h_cpu[v], 1e-4f));
    printf("  [PASS] test_edge_d15\n");
}

static void test_edge_zero_hidden() {
    printf("\n--- Edge: all-zero hidden (D=32, V=16) ---\n");
    const int D = 32, V = 16;

    std::vector<float> h_hidden(D, 0.0f);
    std::vector<float> h_embed(V * D);

    rand_fill(h_embed, -1.0f, 1.0f);

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto h_gpu = to_cpu<float>(lm_head_logits(hidden, embed));

    // All logits must be exactly 0 when hidden is all zeros
    for (int v = 0; v < V; ++v)
        CHECK("zero_hidden:exact_zero", h_gpu[v] == 0.0f);
    printf("  [PASS] test_edge_zero_hidden\n");
}

static void test_edge_zero_embed() {
    printf("\n--- Edge: all-zero embed (D=32, V=16) ---\n");
    const int D = 32, V = 16;

    std::vector<float> h_hidden(D);
    std::vector<float> h_embed(V * D, 0.0f);

    rand_fill(h_hidden, -1.0f, 1.0f);

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto h_gpu = to_cpu<float>(lm_head_logits(hidden, embed));

    // All logits must be exactly 0 when embed is all zeros
    for (int v = 0; v < V; ++v)
        CHECK("zero_embed:exact_zero", h_gpu[v] == 0.0f);
    printf("  [PASS] test_edge_zero_embed\n");
}

static void test_edge_negative_logits() {
    printf("\n--- Edge: all-negative logits (D=32, V=16) ---\n");
    const int D = 32, V = 16;

    // Use negative hidden and positive embed, producing all-negative dot products
    std::vector<float> h_hidden(D);
    std::vector<float> h_embed(V * D);

    for (int d = 0; d < D; ++d)    h_hidden[d] = -1.0f;
    for (int i = 0; i < V * D; ++i) h_embed[i] = 1.0f;

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto h_gpu = to_cpu<float>(lm_head_logits(hidden, embed));
    auto h_cpu = cpu_lm_head(h_hidden.data(), D, h_embed.data(), V);

    // All results should be -32.0 (sum of -1*1 for D=32)
    for (int v = 0; v < V; ++v) {
        CHECK("neg:all_negative", h_gpu[v] < 0.0f);
        CHECK("neg:exact_neg_D", approx(h_gpu[v], (float)(-D), 1e-5f));
    }

    // Verify GPU result agrees with CPU reference
    for (int v = 0; v < V; ++v)
        CHECK("neg:match_cpu", approx(h_gpu[v], h_cpu[v], 1e-5f));

    printf("  [PASS] test_edge_negative_logits\n");
}

static void test_edge_vocab_not_divisible() {
    printf("\n--- Edge: vocab=500 (not divisible by 256), D=32 ---\n");
    const int D = 32, V = 500;

    std::vector<float> h_hidden(D);
    std::vector<float> h_embed(V * D);

    rand_fill(h_hidden, -1.0f, 1.0f);
    rand_fill(h_embed, -1.0f, 1.0f);

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto h_gpu = to_cpu<float>(lm_head_logits(hidden, embed));
    auto h_cpu = cpu_lm_head(h_hidden.data(), D, h_embed.data(), V);

    // Verify all 500 entries
    int bad = 0;
    float max_err = 0.0f;
    for (int v = 0; v < V; ++v) {
        float err = fabsf(h_gpu[v] - h_cpu[v]);
        if (err > max_err) max_err = err;
        if (err >= 1e-4f) bad++;
    }
    CHECK("vocab500:all_approx", bad == 0);
    printf("  max error: %e  bad=%d\n", max_err, bad);

    // Verify that the "tail" entries (v=256..499) are correct — these
    // are handled by threads in the second strided round
    float max_tail_err = 0.0f;
    for (int v = 256; v < V; ++v) {
        float err = fabsf(h_gpu[v] - h_cpu[v]);
        if (err > max_tail_err) max_tail_err = err;
    }
    printf("  max tail error (v>=256): %e\n", max_tail_err);

    printf("  [PASS] test_edge_vocab_not_divisible\n");
}

static void test_edge_hidden_2d_view() {
    printf("\n--- Edge: hidden as [1, D] 2-D tensor (view to [D]) ---\n");
    const int D = 20, V = 10;

    std::vector<float> h_hidden(D);
    std::vector<float> h_embed(V * D);

    rand_fill(h_hidden, -1.0f, 1.0f);
    rand_fill(h_embed, -1.0f, 1.0f);

    // Hidden as [1, D] — the lm_head_logits wrapper should handle the view
    Tensor hidden = make_gpu_tensor({1, D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto h_gpu = to_cpu<float>(lm_head_logits(hidden, embed));
    auto h_cpu = cpu_lm_head(h_hidden.data(), D, h_embed.data(), V);

    for (int v = 0; v < V; ++v)
        CHECK("hidden_2d:approx", approx(h_gpu[v], h_cpu[v], 1e-4f));
    printf("  [PASS] test_edge_hidden_2d_view\n");
}

// ===========================================================================
// SECTION 5: Greedy consistency
// ===========================================================================
static void test_greedy_consistency() {
    printf("\n=== SECTION 5: Greedy Consistency ===\n");

    const int D = 64, V = 256;

    std::vector<float> h_hidden(D);
    std::vector<float> h_embed(V * D);

    rand_fill(h_hidden, -1.0f, 1.0f);
    rand_fill(h_embed, -1.0f, 1.0f);

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    auto h_gpu = to_cpu<float>(lm_head_logits(hidden, embed));
    auto h_cpu = cpu_lm_head(h_hidden.data(), D, h_embed.data(), V);

    // Greedy: argmax (temperature=0)
    auto argmax_gpu = std::max_element(h_gpu.begin(), h_gpu.end()) - h_gpu.begin();
    auto argmax_cpu = std::max_element(h_cpu.begin(), h_cpu.end()) - h_cpu.begin();

    CHECK("greedy:same_token", argmax_gpu == argmax_cpu);
    printf("  GPU argmax: %d (logit=%.6f)\n", (int)argmax_gpu, h_gpu[argmax_gpu]);
    printf("  CPU argmax: %d (logit=%.6f)\n", (int)argmax_cpu, h_cpu[argmax_cpu]);

    // Also verify that the sorted order of top-5 is preserved
    struct IdxVal { int idx; float val; };
    std::vector<IdxVal> gpu_sorted(V), cpu_sorted(V);
    for (int v = 0; v < V; ++v) {
        gpu_sorted[v] = {v, h_gpu[v]};
        cpu_sorted[v] = {v, h_cpu[v]};
    }
    auto cmp = [](const IdxVal& a, const IdxVal& b) { return a.val > b.val; };
    std::sort(gpu_sorted.begin(), gpu_sorted.end(), cmp);
    std::sort(cpu_sorted.begin(), cpu_sorted.end(), cmp);

    // Top-5 should match in order
    for (int k = 0; k < 5; ++k) {
        char buf[64];
        snprintf(buf, sizeof(buf), "greedy:top5_idx_%d", k);
        CHECK(buf, gpu_sorted[k].idx == cpu_sorted[k].idx);
    }

    printf("  [PASS] test_greedy_consistency\n");
}

// ===========================================================================
// SECTION 6: Performance benchmark
// ===========================================================================
static void test_performance() {
    printf("\n=== SECTION 6: Performance (D=896, V=151936) ===\n");

    const int D = 896, V = 151936;
    const size_t hidden_bytes = D * sizeof(float);
    const size_t embed_bytes  = (size_t)V * D * sizeof(float);

    printf("  hidden: %d floats (%zu KB)\n", D, hidden_bytes / 1024);
    printf("  embed:  %d x %d = %zu floats (%.0f MB)\n",
           V, D, (size_t)V * D, embed_bytes / (1024.0 * 1024.0));

    std::vector<float> h_hidden(D);
    std::vector<float> h_embed(V * D);

    rand_fill(h_hidden, -1.0f, 1.0f);
    rand_fill(h_embed, -1.0f, 1.0f);

    Tensor hidden = make_gpu_tensor({D}, h_hidden);
    Tensor embed  = make_gpu_tensor({V, D}, h_embed);

    // Warmup: 3 iterations
    for (int i = 0; i < 3; ++i) {
        auto tmp = lm_head_logits(hidden, embed);
        cudaDeviceSynchronize();
    }

    // Timed: 10 iterations
    const int iters = 10;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        auto tmp = lm_head_logits(hidden, embed);
        cudaDeviceSynchronize();
    }
    auto t1 = std::chrono::steady_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double avg_us = total_us / (double)iters;

    // Compute bandwidth utilisation
    // Reads:  embed (V*D*4 bytes) + hidden (D*4 bytes)
    // Writes: logits (V*4 bytes)
    // Dominant term: embed read = 151936 * 896 * 4 = ~544 MB
    double bytes_read  = (double)((size_t)V * D + D) * sizeof(float);
    double bytes_wrote = (double)V * sizeof(float);
    double total_bytes = bytes_read + bytes_wrote;
    double bw_gb_s = total_bytes / (avg_us * 1e-6) / 1e9;

    // FLOPs: 2 * V * D (multiply-add per element)
    double flops = 2.0 * V * D;
    double tflops = flops / (avg_us * 1e-6) / 1e12;

    printf("\n  GPU kernel (avg over %d iters):\n", iters);
    printf("    wall time:    %.1f us  (%.3f ms)\n", avg_us, avg_us / 1000.0);
    printf("    bandwidth:    %.1f GB/s\n", bw_gb_s);
    printf("    throughput:   %.3f TFLOPS\n", tflops);
    printf("    data moved:   %.1f MB/kernel\n", total_bytes / (1024.0 * 1024.0));

    // Theoretical comparison
    printf("\n  Comparison (A5000 reference):\n");
    printf("    expected GPU: ~780 us  (544 MB / 700 GB/s)\n");
    printf("    old CPU path: ~36000 us (21 ms PCIe + 15 ms compute)\n");
    if (avg_us < 2000.0)
        printf("    => GPU is within expectations (memory-bound kernel)\n");
    else
        printf("    => GPU slower than expected (check kernel launch overhead)\n");

    // Sanity performance check: must complete under 100 ms on any real GPU
    CHECK("perf:under_100ms", avg_us < 100000.0);

    printf("  [PASS] test_performance\n");
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    printf("LightLLM: lm_head kernel correctness tests\n");
    printf("==========================================\n\n");
    fflush(stdout);

    // Seed RNG for reproducibility
    srand(42);

    // Section 1: Small correctness
    test_small_correctness();
    fflush(stdout);

    // Section 2: Medium correctness
    test_medium_correctness();
    fflush(stdout);

    // Section 3: Full-scale correctness
    test_full_scale();
    fflush(stdout);

    // Section 4: Edge cases
    printf("\n=== SECTION 4: Edge Cases ===\n");
    test_edge_d1_v5();               fflush(stdout);
    test_edge_v1_d8();               fflush(stdout);
    test_edge_d7();                  fflush(stdout);
    test_edge_d15();                 fflush(stdout);
    test_edge_zero_hidden();         fflush(stdout);
    test_edge_zero_embed();          fflush(stdout);
    test_edge_negative_logits();     fflush(stdout);
    test_edge_vocab_not_divisible(); fflush(stdout);
    test_edge_hidden_2d_view();      fflush(stdout);

    // Section 5: Greedy consistency
    test_greedy_consistency();
    fflush(stdout);

    // Section 6: Performance
    test_performance();
    fflush(stdout);

    // Summary
    printf("\n==========================================\n");
    printf("%s: %d passed, %d failed\n",
           failed ? "SOME FAILURES" : "ALL PASSED", passed, failed);
    return failed ? 1 : 0;
}
