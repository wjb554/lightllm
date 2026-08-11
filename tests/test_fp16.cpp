#include "lightllm/tensor.h"
#include "lightllm/ops/softmax.h"
#include "lightllm/ops/embedding.h"
#include "lightllm/ops/norm.h"
#include "lightllm/ops/rope.h"
#include "lightllm/ops/elementwise.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace lightllm;
using namespace lightllm::ops;

// ============================================================================
// Helpers
// ============================================================================

/// Simple 32-bit LCG for deterministic pseudo-random floats.
/// Generates values in [0, 1).
static uint32_t lcg_state = 12345;
static void lcg_seed(uint32_t s) { lcg_state = s; }
static float lcg_randf() {
    lcg_state = lcg_state * 1103515245 + 12345;
    return static_cast<float>(lcg_state & 0x7FFFFFFF) / 2147483648.0f;
}

/// Download a GPU tensor into a host vector.
template<typename T>
static std::vector<T> to_cpu(const Tensor& t) {
    std::vector<T> v(t.numel());
    t.copy_to(v.data(), t.nbytes());
    return v;
}

/// Allocate an F32 GPU tensor and upload float data from host.
static Tensor gpu_f32(const std::vector<int>& shape,
                       const std::vector<float>& data) {
    Tensor t(shape, DType::F32, Device::CUDA);
    t.copy_from(data.data(), data.size() * sizeof(float));
    return t;
}

/// Allocate an F16 GPU tensor, convert float data to half, upload.
static Tensor gpu_f16(const std::vector<int>& shape,
                       const std::vector<float>& data) {
    Tensor t(shape, DType::F16, Device::CUDA);
    std::vector<half> h(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        h[i] = __float2half(data[i]);
    t.copy_from(h.data(), h.size() * sizeof(half));
    return t;
}

/// Maximum absolute error: half vector vs float reference.
static float max_err_f16_vs_f32(const std::vector<half>& f16,
                                 const std::vector<float>& f32) {
    float m = 0.0f;
    for (size_t i = 0; i < f16.size(); ++i) {
        float e = fabsf(static_cast<float>(f16[i]) - f32[i]);
        if (e > m) m = e;
    }
    return m;
}

/// Maximum relative error (skip near-zero values).
static float max_rel_err(const std::vector<float>& orig,
                          const std::vector<float>& recovered) {
    float m = 0.0f;
    for (size_t i = 0; i < orig.size(); ++i) {
        if (fabsf(orig[i]) < 1e-8f) continue;
        float r = fabsf(recovered[i] - orig[i]) / fabsf(orig[i]);
        if (r > m) m = r;
    }
    return m;
}

// ============================================================================
// Test 1: softmax fp16 vs fp32
//
// Create identical random values in fp32 and fp16 tensors, run softmax
// on both, compare.  Softmax accumulates in float internally so fp16
// output should closely track fp32.
// ============================================================================
static bool test_softmax_fp16_vs_fp32() {
    const int R = 4, C = 64, N = R * C;
    lcg_seed(42);
    std::vector<float> data(N);
    for (int i = 0; i < N; ++i)
        data[i] = lcg_randf() * 10.0f - 5.0f;  // [-5, 5]

    Tensor x_f32 = gpu_f32({R, C}, data);
    Tensor x_f16 = gpu_f16({R, C}, data);

    Tensor out_f32 = softmax(x_f32);
    Tensor out_f16 = softmax(x_f16);
    cudaDeviceSynchronize();

    auto ref = to_cpu<float>(out_f32);
    auto res = to_cpu<half>(out_f16);

    float err = max_err_f16_vs_f32(res, ref);
    printf("  max_err=%.6f", err);
    return err < 0.01f;
}

// ============================================================================
// Test 2: embedding fp16 vs fp32
//
// Create fp16 and fp32 embedding tables with the same values, gather
// identical token indices from both, compare outputs.
// ============================================================================
static bool test_embedding_fp16_vs_fp32() {
    const int VOCAB = 100, HIDDEN = 32, N_TOK = 16;
    lcg_seed(55);
    std::vector<float> wdata(VOCAB * HIDDEN);
    for (size_t i = 0; i < wdata.size(); ++i)
        wdata[i] = lcg_randf() * 2.0f - 1.0f;

    Tensor w_f32 = gpu_f32({VOCAB, HIDDEN}, wdata);
    Tensor w_f16 = gpu_f16({VOCAB, HIDDEN}, wdata);

    // Token ids: {0, 7, 14, 21, ..., 7*15 % 100}
    std::vector<int> ids(N_TOK);
    for (int i = 0; i < N_TOK; ++i)
        ids[i] = (i * 7) % VOCAB;
    Tensor tok_ids({N_TOK}, DType::I32, Device::CUDA);
    tok_ids.copy_from(ids.data(), N_TOK * sizeof(int));

    Tensor emb_f32 = embedding(tok_ids, w_f32);
    Tensor emb_f16 = embedding(tok_ids, w_f16);
    cudaDeviceSynchronize();

    auto ref = to_cpu<float>(emb_f32);
    auto res = to_cpu<half>(emb_f16);

    float err = max_err_f16_vs_f32(res, ref);
    printf("  max_err=%.6f", err);
    return err < 0.001f;
}

// ============================================================================
// Test 3: RMSNorm fp16 vs fp32
//
// RMSNorm kernel already supports fp16 via DISPATCH_FLOAT_HALF.
// Compare fp16 output against fp32 reference.
// ============================================================================
static bool test_norm_fp16_vs_fp32() {
    const int N = 4, D = 64;
    lcg_seed(33);
    std::vector<float> xdata(N * D);
    for (size_t i = 0; i < xdata.size(); ++i)
        xdata[i] = lcg_randf() * 4.0f - 2.0f;
    std::vector<float> wdata(D, 1.0f);  // unit weight

    Tensor x_f32 = gpu_f32({N, D}, xdata);
    Tensor w_f32 = gpu_f32({D}, wdata);
    Tensor x_f16 = gpu_f16({N, D}, xdata);
    Tensor w_f16 = gpu_f16({D}, wdata);

    Tensor norm_f32 = rms_norm(x_f32, w_f32);
    Tensor norm_f16 = rms_norm(x_f16, w_f16);
    cudaDeviceSynchronize();

    auto ref = to_cpu<float>(norm_f32);
    auto res = to_cpu<half>(norm_f16);

    float err = max_err_f16_vs_f32(res, ref);
    printf("  max_err=%.6f", err);
    return err < 0.01f;
}

// ============================================================================
// Test 4: RoPE fp16 vs fp32
//
// RoPE kernel already supports fp16 via DISPATCH_FLOAT_HALF.
// Apply the same rotation to fp32 and fp16 query tensors, compare.
// ============================================================================
static bool test_rope_fp16_vs_fp32() {
    const int T = 2, H = 4, D = 64, N = T * H * D;
    lcg_seed(77);
    std::vector<float> qdata(N);
    for (int i = 0; i < N; ++i)
        qdata[i] = lcg_randf() * 2.0f - 1.0f;

    Tensor q_f32 = gpu_f32({T, H, D}, qdata);
    Tensor q_f16 = gpu_f16({T, H, D}, qdata);

    auto [cos, sin] = make_rope_cos_sin(T, D, 10000.0f);

    rope(q_f32, nullptr, cos, sin);
    rope(q_f16, nullptr, cos, sin);
    cudaDeviceSynchronize();

    auto ref = to_cpu<float>(q_f32);
    auto res = to_cpu<half>(q_f16);

    float err = max_err_f16_vs_f32(res, ref);
    printf("  max_err=%.6f", err);
    return err < 0.02f;
}

// ============================================================================
// Test 5: elementwise fp16 (add + mul)
//
// add and mul already support fp16 via DISPATCH_FLOAT_TYPES.
// Verify fp16 results match fp32 within tight tolerance.
// ============================================================================
static bool test_elementwise_fp16() {
    const int N = 256;
    lcg_seed(11);
    std::vector<float> va(N), vb(N);
    for (int i = 0; i < N; ++i) {
        va[i] = lcg_randf() * 2.0f - 1.0f;
        vb[i] = lcg_randf() * 2.0f - 1.0f;
    }

    Tensor a_f32 = gpu_f32({N}, va);
    Tensor b_f32 = gpu_f32({N}, vb);
    Tensor a_f16 = gpu_f16({N}, va);
    Tensor b_f16 = gpu_f16({N}, vb);

    // ---- add ----
    Tensor add_f32 = add(a_f32, b_f32);
    Tensor add_f16 = add(a_f16, b_f16);
    float err_add = max_err_f16_vs_f32(to_cpu<half>(add_f16),
                                        to_cpu<float>(add_f32));
    if (err_add >= 0.001f) {
        printf("  add max_err=%.6f", err_add);
        return false;
    }

    // ---- mul ----
    Tensor mul_f32 = mul(a_f32, b_f32);
    Tensor mul_f16 = mul(a_f16, b_f16);
    float err_mul = max_err_f16_vs_f32(to_cpu<half>(mul_f16),
                                        to_cpu<float>(mul_f32));
    if (err_mul >= 0.001f) {
        printf("  mul max_err=%.6f", err_mul);
        return false;
    }

    printf("  add_err=%.6f mul_err=%.6f", err_add, err_mul);
    return true;
}

// ============================================================================
// Test 6: fp16 roundtrip (F32 -> F16 -> F32)
//
// Convert float values to half (via GPU F16 tensor) and back.
// Verify relative error is within fp16 precision (~0.1%).
// ============================================================================
static bool test_fp16_roundtrip() {
    const int N = 256;
    lcg_seed(99);
    std::vector<float> original(N);
    for (int i = 0; i < N; ++i)
        original[i] = lcg_randf() * 2.0f - 1.0f;

    // F32 -> F16 -> back to F32 via GPU roundtrip
    Tensor f16_tensor = gpu_f16({N}, original);
    auto half_data = to_cpu<half>(f16_tensor);
    std::vector<float> recovered(N);
    for (int i = 0; i < N; ++i)
        recovered[i] = static_cast<float>(half_data[i]);

    float rel_err = max_rel_err(original, recovered);

    // Also check max absolute error for very small values
    float abs_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float e = fabsf(recovered[i] - original[i]);
        if (e > abs_err) abs_err = e;
    }

    printf("  rel_err=%.6f abs_err=%.6f", rel_err, abs_err);
    return rel_err < 0.002f;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("=== FP16 Correctness Tests ===\n\n");

    struct Test { const char* name; bool (*fn)(); };
    Test tests[] = {
        {"test_softmax_fp16_vs_fp32",    test_softmax_fp16_vs_fp32},
        {"test_embedding_fp16_vs_fp32",  test_embedding_fp16_vs_fp32},
        {"test_norm_fp16_vs_fp32",       test_norm_fp16_vs_fp32},
        {"test_rope_fp16_vs_fp32",       test_rope_fp16_vs_fp32},
        {"test_elementwise_fp16",        test_elementwise_fp16},
        {"test_fp16_roundtrip",          test_fp16_roundtrip},
    };

    int passed = 0, failed = 0;
    for (auto& t : tests) {
        printf("  %-40s ", t.name);
        fflush(stdout);
        try {
            if (t.fn()) {
                printf("  PASS\n");
                ++passed;
            } else {
                printf("  FAIL\n");
                ++failed;
            }
        } catch (const std::exception& e) {
            printf("  CRASH: %s\n", e.what());
            ++failed;
        }
    }

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
