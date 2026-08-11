/// LightLLM — Attention FP16 Correctness Tests
///
/// Verifies that fp16 paged attention (decode and prefill) produces results
/// indistinguishable from fp32 attention.
///
/// Tests:
///   1. Decode baseline (D=64,  seq_len=32)  — scalar or Float4 path
///   2. Decode v2       (D=128, seq_len=32)  — Float4 path (D%8==0 for half)
///   3. Decode Split-K  (D=64,  seq_len=512) — Split-K path
///   4. Prefill paged   (P=32,  D=64)        — direct prefill kernel
///   5. Prefill Flash   (P=128, D=64)        — tiled FlashAttention path
///   6. Gradient-free   (D=128)              — no NaN/Inf in output

#include "lightllm/tensor.h"
#include "lightllm/kv_cache/block_allocator.h"
#include "lightllm/kv_cache/paged_attention.h"

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace lightllm;
using namespace lightllm::kv_cache;

// ============================================================================
// LCG pseudo-random generator (deterministic, portable)
// ============================================================================
static uint32_t rng_state = 12345;

static void rng_seed(uint32_t s) { rng_state = s; }

static float rng_randf() {
    rng_state = rng_state * 1103515245 + 12345;
    return static_cast<float>(rng_state & 0x7FFFFFFF) / 2147483648.0f;
}

// ============================================================================
// Test harness
// ============================================================================
static int passed = 0;
static int failed = 0;

static void CHECK(const char* name, bool cond) {
    if (cond) {
        passed++;
    } else {
        failed++;
        std::fprintf(stderr, "  FAIL: %s\n", name);
    }
}

// ============================================================================
// Math helpers (CPU, float)
// ============================================================================

static float max_abs_error(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = std::fabs(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

static float dot(const float* a, const float* b, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; i++) d += a[i] * b[i];
    return d;
}

static float l2_norm(const float* a, int n) {
    return std::sqrt(dot(a, a, n));
}

static float cosine_similarity(const float* a, const float* b, int n) {
    float na = l2_norm(a, n);
    float nb = l2_norm(b, n);
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return dot(a, b, n) / (na * nb);
}

// Check all values are finite (no NaN, no Inf)
static bool all_finite(const float* a, int n) {
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i])) return false;
    }
    return true;
}

// ============================================================================
// Data generation & conversion helpers
// ============================================================================

/// Fill host vector with uniform random floats in [-1, 1).
static std::vector<float> rand_vec(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; i++)
        v[i] = rng_randf() * 2.0f - 1.0f;
    return v;
}

/// Convert float vector to half vector.
static std::vector<half> f32_to_f16(const std::vector<float>& data) {
    std::vector<half> h(data.size());
    for (size_t i = 0; i < data.size(); i++)
        h[i] = __float2half(data[i]);
    return h;
}

/// Convert half vector to float vector.
static std::vector<float> f16_to_f32(const std::vector<half>& data) {
    std::vector<float> f(data.size());
    for (size_t i = 0; i < data.size(); i++)
        f[i] = __half2float(data[i]);
    return f;
}

// ============================================================================
// GPU tensor helpers
// ============================================================================

/// Create GPU tensor from host float data.
static Tensor gpu_f32(const std::vector<int>& shape,
                       const std::vector<float>& data) {
    Tensor t(shape, DType::F32, Device::CUDA);
    t.copy_from(data.data(), data.size() * sizeof(float));
    return t;
}

/// Create GPU tensor from host half data.
static Tensor gpu_f16(const std::vector<int>& shape,
                       const std::vector<half>& data) {
    Tensor t(shape, DType::F16, Device::CUDA);
    t.copy_from(data.data(), data.size() * sizeof(half));
    return t;
}

/// Create GPU I32 tensor from host int data.
static Tensor gpu_i32(const std::vector<int>& shape,
                       const std::vector<int>& data) {
    Tensor t(shape, DType::I32, Device::CUDA);
    t.copy_from(data.data(), data.size() * sizeof(int));
    return t;
}

/// Download F32 GPU tensor to host float vector.
static std::vector<float> to_cpu_f32(const Tensor& t) {
    std::vector<float> v(t.numel());
    t.copy_to(v.data(), t.nbytes());
    return v;
}

/// Download F16 GPU tensor and convert to host float vector.
static std::vector<float> to_cpu_f16_as_f32(const Tensor& t) {
    std::vector<half> h(t.numel());
    t.copy_to(h.data(), t.nbytes());
    return f16_to_f32(h);
}

// ============================================================================
// Shared test infrastructure
// ============================================================================

/// Parameters for one attention test configuration.
struct AttnTestConfig {
    int num_tokens;   // number of decode tokens (1 for typical decode)
    int Hq;           // query heads
    int Hkv;          // key/value heads
    int D;            // head dimension
    int seq_len;      // total KV sequence length
    int block_size;   // tokens per block (default 16)
    int num_blocks;   // total blocks in allocator pool
};

/// Result of running one attention test variant.
struct AttnResult {
    std::vector<float> output;  // output as host float
};

/// Allocate blocks, scatter K/V into blocks, return block_table as GPU tensor.
///
/// Returns a pair: (block_table_gpu, block_ids_host).
/// Caller must release blocks after the attention call completes.
static std::pair<Tensor, std::vector<int>>
setup_kv_blocks(BlockAllocator& alloc,
                const float* k_host, const float* v_host,
                int seq_len, int Hkv, int D)
{
    int block_size = alloc.block_size();
    int n_blocks = (seq_len + block_size - 1) / block_size;
    int pool_blocks = alloc.num_blocks();

    // Allocate blocks
    std::vector<int> block_ids;
    for (int i = 0; i < n_blocks; i++) {
        int bid = alloc.allocate(/*token_hash=*/0);
        if (bid < 0) {
            std::fprintf(stderr, "  FAIL: block pool exhausted (need %d, have %d)\n",
                         n_blocks, pool_blocks);
            std::exit(1);
        }
        block_ids.push_back(bid);
    }

    // Upload block_table to GPU
    Tensor bt_gpu = gpu_i32({n_blocks}, block_ids);

    // Scatter K/V into blocks
    DType dtype = alloc.kv_dtype();
    if (dtype == DType::F16) {
        auto k_half = f32_to_f16(
            std::vector<float>(k_host, k_host + seq_len * Hkv * D));
        auto v_half = f32_to_f16(
            std::vector<float>(v_host, v_host + seq_len * Hkv * D));
        Tensor k_gpu = gpu_f16({seq_len, Hkv, D}, k_half);
        Tensor v_gpu = gpu_f16({seq_len, Hkv, D}, v_half);
        scatter_prefill_kv_gpu(k_gpu, v_gpu, /*start_pos=*/0, seq_len,
                               bt_gpu.data<const int>(), n_blocks, alloc);
    } else {
        Tensor k_gpu = gpu_f32({seq_len, Hkv, D},
            std::vector<float>(k_host, k_host + seq_len * Hkv * D));
        Tensor v_gpu = gpu_f32({seq_len, Hkv, D},
            std::vector<float>(v_host, v_host + seq_len * Hkv * D));
        scatter_prefill_kv_gpu(k_gpu, v_gpu, /*start_pos=*/0, seq_len,
                               bt_gpu.data<const int>(), n_blocks, alloc);
    }
    cudaDeviceSynchronize();

    return {std::move(bt_gpu), std::move(block_ids)};
}

/// Run decode paged_attention and return host float output.
static std::vector<float>
run_decode_attention(const float* q_host, int num_tokens, int Hq, int D,
                     const int* block_table_dev, int n_blocks,
                     const int* seq_lens_dev,
                     BlockAllocator& alloc)
{
    DType dtype = alloc.kv_dtype();
    if (dtype == DType::F16) {
        auto q_half = f32_to_f16(
            std::vector<float>(q_host, q_host + num_tokens * Hq * D));
        Tensor q_gpu = gpu_f16({num_tokens, Hq, D}, q_half);
        Tensor out = paged_attention(q_gpu, block_table_dev, n_blocks,
                                     seq_lens_dev, alloc);
        cudaDeviceSynchronize();
        return to_cpu_f16_as_f32(out);
    } else {
        Tensor q_gpu = gpu_f32({num_tokens, Hq, D},
            std::vector<float>(q_host, q_host + num_tokens * Hq * D));
        Tensor out = paged_attention(q_gpu, block_table_dev, n_blocks,
                                     seq_lens_dev, alloc);
        cudaDeviceSynchronize();
        return to_cpu_f32(out);
    }
}

/// Run prefill paged_attention and return host float output.
static std::vector<float>
run_prefill_attention(const float* q_host, int P, int Hq, int D,
                      const int* block_table_dev, int n_blocks,
                      const int* seq_len_dev,
                      BlockAllocator& alloc, bool causal)
{
    DType dtype = alloc.kv_dtype();
    if (dtype == DType::F16) {
        auto q_half = f32_to_f16(
            std::vector<float>(q_host, q_host + P * Hq * D));
        Tensor q_gpu = gpu_f16({P, Hq, D}, q_half);
        Tensor out = prefill_paged_attention(q_gpu, block_table_dev,
            n_blocks, seq_len_dev, alloc, causal);
        cudaDeviceSynchronize();
        return to_cpu_f16_as_f32(out);
    } else {
        Tensor q_gpu = gpu_f32({P, Hq, D},
            std::vector<float>(q_host, q_host + P * Hq * D));
        Tensor out = prefill_paged_attention(q_gpu, block_table_dev,
            n_blocks, seq_len_dev, alloc, causal);
        cudaDeviceSynchronize();
        return to_cpu_f32(out);
    }
}

// ============================================================================
// Test 1: Paged Decode Baseline (D=64, seq_len=32)
//
// Small decode attention: 1 token, 2 heads, D=64, 32 KV tokens.
// Verifies fp16 scalar/Float4 decode path matches fp32.
// ============================================================================
static bool test_paged_decode_baseline_fp16() {
    AttnTestConfig cfg = {1, 2, 2, 64, 32, 16, 8};
    int n_blocks = (cfg.seq_len + cfg.block_size - 1) / cfg.block_size; // 2

    rng_seed(42);

    // Generate random K/V and Q data
    auto k_data = rand_vec(cfg.seq_len * cfg.Hkv * cfg.D);
    auto v_data = rand_vec(cfg.seq_len * cfg.Hkv * cfg.D);
    auto q_data = rand_vec(cfg.num_tokens * cfg.Hq * cfg.D);

    // ---- F32 path ----
    BlockAllocator alloc_f32(cfg.num_blocks, cfg.block_size, cfg.Hkv, cfg.D,
                             PrefixCachePolicy::Hash, DType::F32);
    auto [bt_f32, bids_f32] = setup_kv_blocks(alloc_f32,
        k_data.data(), v_data.data(), cfg.seq_len, cfg.Hkv, cfg.D);
    Tensor seq_lens_gpu = gpu_i32({1}, {cfg.seq_len});
    auto ref = run_decode_attention(q_data.data(), cfg.num_tokens,
        cfg.Hq, cfg.D, bt_f32.data<const int>(), n_blocks,
        seq_lens_gpu.data<const int>(), alloc_f32);

    // ---- F16 path ----
    BlockAllocator alloc_f16(cfg.num_blocks, cfg.block_size, cfg.Hkv, cfg.D,
                             PrefixCachePolicy::Hash, DType::F16);
    auto [bt_f16, bids_f16] = setup_kv_blocks(alloc_f16,
        k_data.data(), v_data.data(), cfg.seq_len, cfg.Hkv, cfg.D);
    auto res = run_decode_attention(q_data.data(), cfg.num_tokens,
        cfg.Hq, cfg.D, bt_f16.data<const int>(), n_blocks,
        seq_lens_gpu.data<const int>(), alloc_f16);

    // ---- Compare ----
    float max_err = max_abs_error(ref.data(), res.data(), (int)ref.size());
    float cos_sim = cosine_similarity(ref.data(), res.data(), (int)ref.size());
    std::printf("  decode_baseline D=%d: max_err=%.6f cos_sim=%.6f\n", cfg.D, max_err, cos_sim);

    CHECK("decode_baseline: max_err < 0.01", max_err < 0.01f);
    CHECK("decode_baseline: cos_sim > 0.999", cos_sim > 0.999f);

    // Cleanup
    for (int bid : bids_f32) alloc_f32.release(bid);
    for (int bid : bids_f16) alloc_f16.release(bid);

    return max_err < 0.01f && cos_sim > 0.999f;
}

// ============================================================================
// Test 2: Paged Decode V2 (D=128, seq_len=32)
//
// D=128 triggers the Float4 vectorized path (D%8==0 for half).
// Verifies fp16 Float4 path matches fp32 Float4 path.
// ============================================================================
static bool test_paged_decode_v2_fp16() {
    AttnTestConfig cfg = {1, 2, 2, 128, 32, 16, 8};
    int n_blocks = (cfg.seq_len + cfg.block_size - 1) / cfg.block_size; // 2

    rng_seed(99);

    auto k_data = rand_vec(cfg.seq_len * cfg.Hkv * cfg.D);
    auto v_data = rand_vec(cfg.seq_len * cfg.Hkv * cfg.D);
    auto q_data = rand_vec(cfg.num_tokens * cfg.Hq * cfg.D);

    // F32
    BlockAllocator alloc_f32(cfg.num_blocks, cfg.block_size, cfg.Hkv, cfg.D,
                             PrefixCachePolicy::Hash, DType::F32);
    auto [bt_f32, bids_f32] = setup_kv_blocks(alloc_f32,
        k_data.data(), v_data.data(), cfg.seq_len, cfg.Hkv, cfg.D);
    Tensor seq_lens_gpu = gpu_i32({1}, {cfg.seq_len});
    auto ref = run_decode_attention(q_data.data(), cfg.num_tokens,
        cfg.Hq, cfg.D, bt_f32.data<const int>(), n_blocks,
        seq_lens_gpu.data<const int>(), alloc_f32);

    // F16
    BlockAllocator alloc_f16(cfg.num_blocks, cfg.block_size, cfg.Hkv, cfg.D,
                             PrefixCachePolicy::Hash, DType::F16);
    auto [bt_f16, bids_f16] = setup_kv_blocks(alloc_f16,
        k_data.data(), v_data.data(), cfg.seq_len, cfg.Hkv, cfg.D);
    auto res = run_decode_attention(q_data.data(), cfg.num_tokens,
        cfg.Hq, cfg.D, bt_f16.data<const int>(), n_blocks,
        seq_lens_gpu.data<const int>(), alloc_f16);

    float max_err = max_abs_error(ref.data(), res.data(), (int)ref.size());
    float cos_sim = cosine_similarity(ref.data(), res.data(), (int)ref.size());
    std::printf("  decode_v2     D=%d: max_err=%.6f cos_sim=%.6f\n", cfg.D, max_err, cos_sim);

    CHECK("decode_v2: max_err < 0.01", max_err < 0.01f);
    CHECK("decode_v2: cos_sim > 0.999", cos_sim > 0.999f);

    for (int bid : bids_f32) alloc_f32.release(bid);
    for (int bid : bids_f16) alloc_f16.release(bid);

    return max_err < 0.01f && cos_sim > 0.999f;
}

// ============================================================================
// Test 3: Paged Decode Split-K (D=64, seq_len=512)
//
// Long sequence triggers Split-K path (threshold = 256).
// Verifies fp16 Split-K matches fp32 Split-K.
// ============================================================================
static bool test_paged_decode_splitk_fp16() {
    AttnTestConfig cfg = {1, 2, 2, 64, 512, 16, 64};
    // seq_len=512, block_size=16 -> 32 blocks

    rng_seed(777);

    auto k_data = rand_vec(cfg.seq_len * cfg.Hkv * cfg.D);
    auto v_data = rand_vec(cfg.seq_len * cfg.Hkv * cfg.D);
    auto q_data = rand_vec(cfg.num_tokens * cfg.Hq * cfg.D);

    int n_blocks = (cfg.seq_len + cfg.block_size - 1) / cfg.block_size;

    // F32
    BlockAllocator alloc_f32(cfg.num_blocks, cfg.block_size, cfg.Hkv, cfg.D,
                             PrefixCachePolicy::Hash, DType::F32);
    auto [bt_f32, bids_f32] = setup_kv_blocks(alloc_f32,
        k_data.data(), v_data.data(), cfg.seq_len, cfg.Hkv, cfg.D);
    Tensor seq_lens_gpu = gpu_i32({1}, {cfg.seq_len});
    auto ref = run_decode_attention(q_data.data(), cfg.num_tokens,
        cfg.Hq, cfg.D, bt_f32.data<const int>(), n_blocks,
        seq_lens_gpu.data<const int>(), alloc_f32);

    // F16
    BlockAllocator alloc_f16(cfg.num_blocks, cfg.block_size, cfg.Hkv, cfg.D,
                             PrefixCachePolicy::Hash, DType::F16);
    auto [bt_f16, bids_f16] = setup_kv_blocks(alloc_f16,
        k_data.data(), v_data.data(), cfg.seq_len, cfg.Hkv, cfg.D);
    auto res = run_decode_attention(q_data.data(), cfg.num_tokens,
        cfg.Hq, cfg.D, bt_f16.data<const int>(), n_blocks,
        seq_lens_gpu.data<const int>(), alloc_f16);

    float max_err = max_abs_error(ref.data(), res.data(), (int)ref.size());
    float cos_sim = cosine_similarity(ref.data(), res.data(), (int)ref.size());
    std::printf("  decode_splitk D=%d seq=%d: max_err=%.6f cos_sim=%.6f\n",
                cfg.D, cfg.seq_len, max_err, cos_sim);

    // Split-K has more accumulated error — relax tolerance slightly
    CHECK("decode_splitk: max_err < 0.05", max_err < 0.05f);
    CHECK("decode_splitk: cos_sim > 0.995", cos_sim > 0.995f);

    for (int bid : bids_f32) alloc_f32.release(bid);
    for (int bid : bids_f16) alloc_f16.release(bid);

    return max_err < 0.05f && cos_sim > 0.995f;
}

// ============================================================================
// Test 4: Prefill Paged Attention (P=32, D=64)
//
// Short prefill uses direct kernel.  Fp16 vs fp32 comparison.
// ============================================================================
static bool test_prefill_paged_fp16() {
    AttnTestConfig cfg = {1, 2, 2, 64, 32, 16, 8};
    int P = 32; // prefill chunk = all tokens (fresh sequence)
    int Hq = 2, Hkv = 2, D = 64;
    int n_blocks = (P + cfg.block_size - 1) / cfg.block_size; // 2

    rng_seed(123);

    // K/V for the prefill tokens
    auto k_data = rand_vec(P * Hkv * D);
    auto v_data = rand_vec(P * Hkv * D);
    // Q for prefill tokens (different from K/V — they're different projections)
    auto q_data = rand_vec(P * Hq * D);

    // F32
    BlockAllocator alloc_f32(cfg.num_blocks, cfg.block_size, Hkv, D,
                             PrefixCachePolicy::Hash, DType::F32);
    auto [bt_f32, bids_f32] = setup_kv_blocks(alloc_f32,
        k_data.data(), v_data.data(), P, Hkv, D);
    Tensor seq_len_gpu = gpu_i32({1}, {P});
    auto ref = run_prefill_attention(q_data.data(), P, Hq, D,
        bt_f32.data<const int>(), n_blocks,
        seq_len_gpu.data<const int>(), alloc_f32, /*causal=*/true);

    // F16
    BlockAllocator alloc_f16(cfg.num_blocks, cfg.block_size, Hkv, D,
                             PrefixCachePolicy::Hash, DType::F16);
    auto [bt_f16, bids_f16] = setup_kv_blocks(alloc_f16,
        k_data.data(), v_data.data(), P, Hkv, D);
    auto res = run_prefill_attention(q_data.data(), P, Hq, D,
        bt_f16.data<const int>(), n_blocks,
        seq_len_gpu.data<const int>(), alloc_f16, /*causal=*/true);

    float max_err = max_abs_error(ref.data(), res.data(), (int)ref.size());
    float cos_sim = cosine_similarity(ref.data(), res.data(), (int)ref.size());
    std::printf("  prefill_paged P=%d D=%d: max_err=%.6f cos_sim=%.6f\n",
                P, D, max_err, cos_sim);

    CHECK("prefill_paged: max_err < 0.01", max_err < 0.01f);
    CHECK("prefill_paged: cos_sim > 0.999", cos_sim > 0.999f);

    for (int bid : bids_f32) alloc_f32.release(bid);
    for (int bid : bids_f16) alloc_f16.release(bid);

    return max_err < 0.01f && cos_sim > 0.999f;
}

// ============================================================================
// Test 5: Prefill FlashAttention (P=128, D=64)
//
// P > 64 triggers the tiled FlashAttention path.
// Verifies fp16 Flash matches fp32 Flash.
// ============================================================================
static bool test_prefill_flash_fp16() {
    int P = 128, Hq = 2, Hkv = 2, D = 64;
    int block_size = 16, num_blocks = 32;
    int n_blocks = (P + block_size - 1) / block_size; // 8

    rng_seed(456);

    auto k_data = rand_vec(P * Hkv * D);
    auto v_data = rand_vec(P * Hkv * D);
    auto q_data = rand_vec(P * Hq * D);

    // F32
    BlockAllocator alloc_f32(num_blocks, block_size, Hkv, D,
                             PrefixCachePolicy::Hash, DType::F32);
    auto [bt_f32, bids_f32] = setup_kv_blocks(alloc_f32,
        k_data.data(), v_data.data(), P, Hkv, D);
    Tensor seq_len_gpu = gpu_i32({1}, {P});
    auto ref = run_prefill_attention(q_data.data(), P, Hq, D,
        bt_f32.data<const int>(), n_blocks,
        seq_len_gpu.data<const int>(), alloc_f32, /*causal=*/true);

    // F16
    BlockAllocator alloc_f16(num_blocks, block_size, Hkv, D,
                             PrefixCachePolicy::Hash, DType::F16);
    auto [bt_f16, bids_f16] = setup_kv_blocks(alloc_f16,
        k_data.data(), v_data.data(), P, Hkv, D);
    auto res = run_prefill_attention(q_data.data(), P, Hq, D,
        bt_f16.data<const int>(), n_blocks,
        seq_len_gpu.data<const int>(), alloc_f16, /*causal=*/true);

    float max_err = max_abs_error(ref.data(), res.data(), (int)ref.size());
    float cos_sim = cosine_similarity(ref.data(), res.data(), (int)ref.size());
    std::printf("  prefill_flash P=%d D=%d: max_err=%.6f cos_sim=%.6f\n",
                P, D, max_err, cos_sim);

    CHECK("prefill_flash: max_err < 0.01", max_err < 0.01f);
    CHECK("prefill_flash: cos_sim > 0.999", cos_sim > 0.999f);

    for (int bid : bids_f32) alloc_f32.release(bid);
    for (int bid : bids_f16) alloc_f16.release(bid);

    return max_err < 0.01f && cos_sim > 0.999f;
}

// ============================================================================
// Test 6: Attention FP16 Gradient-Free (no NaN, no Inf)
//
// Verify fp16 attention with D=128 does not crash and produces finite values.
// All output elements must be finite.
// ============================================================================
static bool test_attention_fp16_gradient_free() {
    int num_tokens = 1, Hq = 2, Hkv = 2, D = 128;
    int seq_len = 64, block_size = 16, num_blocks = 16;
    int n_blocks = (seq_len + block_size - 1) / block_size; // 4

    rng_seed(888);

    auto k_data = rand_vec(seq_len * Hkv * D);
    auto v_data = rand_vec(seq_len * Hkv * D);
    auto q_data = rand_vec(num_tokens * Hq * D);

    // Test decode
    {
        BlockAllocator alloc(num_blocks, block_size, Hkv, D,
                             PrefixCachePolicy::Hash, DType::F16);
        auto [bt_gpu, bids] = setup_kv_blocks(alloc,
            k_data.data(), v_data.data(), seq_len, Hkv, D);
        Tensor seq_lens_gpu = gpu_i32({1}, {seq_len});
        auto out = run_decode_attention(q_data.data(), num_tokens,
            Hq, D, bt_gpu.data<const int>(), n_blocks,
            seq_lens_gpu.data<const int>(), alloc);

        bool ok = all_finite(out.data(), (int)out.size());
        CHECK("gradient_free/decode_finite", ok);

        for (int bid : bids) alloc.release(bid);
    }

    // Test prefill
    {
        int P = 32;
        auto k_data2 = rand_vec(P * Hkv * D);
        auto v_data2 = rand_vec(P * Hkv * D);
        auto q_data2 = rand_vec(P * Hq * D);

        BlockAllocator alloc(8, block_size, Hkv, D,
                             PrefixCachePolicy::Hash, DType::F16);
        auto [bt_gpu, bids] = setup_kv_blocks(alloc,
            k_data2.data(), v_data2.data(), P, Hkv, D);
        int n_blk = (P + block_size - 1) / block_size;
        Tensor seq_len_gpu = gpu_i32({1}, {P});
        auto out = run_prefill_attention(q_data2.data(), P, Hq, D,
            bt_gpu.data<const int>(), n_blk,
            seq_len_gpu.data<const int>(), alloc, /*causal=*/false);

        bool ok = all_finite(out.data(), (int)out.size());
        CHECK("gradient_free/prefill_finite", ok);

        for (int bid : bids) alloc.release(bid);
    }

    std::printf("  gradient_free D=%d: decode+prefill finite check completed\n", D);
    return true; // all CHECKs inside
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::printf("=== Attention FP16 Correctness Tests ===\n\n");

    struct TestCase {
        const char* name;
        bool (*fn)();
    };

    TestCase tests[] = {
        {"test_paged_decode_baseline_fp16",  test_paged_decode_baseline_fp16},
        {"test_paged_decode_v2_fp16",        test_paged_decode_v2_fp16},
        {"test_paged_decode_splitk_fp16",    test_paged_decode_splitk_fp16},
        {"test_prefill_paged_fp16",          test_prefill_paged_fp16},
        {"test_prefill_flash_fp16",          test_prefill_flash_fp16},
        {"test_attention_fp16_gradient_free",test_attention_fp16_gradient_free},
    };

    int total_passed = 0, total_failed = 0;
    for (auto& tc : tests) {
        std::printf("  %-40s ", tc.name);
        std::fflush(stdout);
        passed = 0;
        failed = 0;

        try {
            tc.fn();
        } catch (const std::exception& e) {
            std::printf("  CRASH: %s\n", e.what());
            failed++;
        }

        total_passed += passed;
        total_failed += failed;
    }

    std::printf("\n%d assertions passed, %d assertions failed (across %d tests)\n",
                total_passed, total_failed,
                (int)(sizeof(tests) / sizeof(tests[0])));
    return total_failed ? 1 : 0;
}
