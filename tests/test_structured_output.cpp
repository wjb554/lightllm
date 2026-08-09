#include "lightllm/ops/sampling.h"
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <set>
#include <cmath>

using namespace lightllm::ops;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static bool is_neg_inf(float v) {
    return std::isinf(v) && v < 0.0f;
}

static int32_t* mask_data(std::vector<int32_t>& m) {
    return m.empty() ? nullptr : m.data();
}

// ---------------------------------------------------------------------------
// TEST 1: mask_apply_basic
// ---------------------------------------------------------------------------
/// 8 logits, mask allows only tokens 3 and 7.
/// Those two must stay unchanged; everything else must become -inf.
bool mask_apply_basic() {
    std::vector<float> logits = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    int vocab_size = 8;

    // Bits 3 and 7 set → (1<<3) | (1<<7) = 8 + 128 = 136
    std::vector<int32_t> mask(1, 0);
    mask[0] = (1u << 3) | (1u << 7);

    apply_grammar_mask(logits.data(), vocab_size, mask.data());

    // Allowed tokens must be unchanged
    if (is_neg_inf(logits[3])) return false;
    if (is_neg_inf(logits[7])) return false;
    if (logits[3] != 0.4f) return false;
    if (logits[7] != 0.8f) return false;

    // Everything else must be -inf
    for (int i = 0; i < vocab_size; ++i) {
        if (i == 3 || i == 7) continue;
        if (!is_neg_inf(logits[i])) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// TEST 2: mask_apply_all_allowed
// ---------------------------------------------------------------------------
/// 16 logits, mask with all 16 low bits set → no token masked.
bool mask_apply_all_allowed() {
    std::vector<float> logits(16);
    for (int i = 0; i < 16; ++i) logits[i] = static_cast<float>(i);
    int vocab_size = 16;

    std::vector<int32_t> mask(1, 0xFFFF);   // bits 0–15 set

    apply_grammar_mask(logits.data(), vocab_size, mask.data());

    for (int i = 0; i < vocab_size; ++i) {
        if (is_neg_inf(logits[i]))          return false;
        if (logits[i] != static_cast<float>(i)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// TEST 3: mask_apply_none_allowed
// ---------------------------------------------------------------------------
/// 16 logits, mask all-zeros → every token becomes -inf.
bool mask_apply_none_allowed() {
    std::vector<float> logits(16);
    for (int i = 0; i < 16; ++i) logits[i] = static_cast<float>(i);
    int vocab_size = 16;

    std::vector<int32_t> mask(1, 0);        // no bit set

    apply_grammar_mask(logits.data(), vocab_size, mask.data());

    for (int i = 0; i < vocab_size; ++i)
        if (!is_neg_inf(logits[i])) return false;
    return true;
}

// ---------------------------------------------------------------------------
// TEST 4: sample_with_mask_deterministic
// ---------------------------------------------------------------------------
/// vocab=100, token 42 has value 100.0 (all others 0.0).
/// Mask allows {40,41,42,43,44}.  Greedy → must always return 42.
bool sample_with_mask_deterministic() {
    int vocab_size = 100;
    std::vector<float> logits(vocab_size, 0.0f);
    logits[42] = 100.0f;

    // 100 tokens → ⌈100/32⌉ = 4 words
    std::vector<int32_t> mask(4, 0);
    for (int t = 40; t <= 44; ++t) {
        int word = t / 32;
        int bit  = t % 32;
        mask[word] |= (1u << bit);
    }

    SamplingParams params;
    params.temperature = 0.0f;              // greedy
    std::mt19937 rng(42);

    for (int i = 0; i < 100; ++i) {
        int tok = sample(logits.data(), vocab_size, params, rng, mask.data());
        if (tok != 42) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// TEST 5: sample_with_mask_excludes_token
// ---------------------------------------------------------------------------
/// vocab=100, token 0 has value 1000.0 (overwhelming argmax).
/// Mask BLOCKS token 0 (bit 0 = 0).  Greedy must NOT return 0
/// but instead the best *allowed* token.
bool sample_with_mask_excludes_token() {
    int vocab_size = 100;
    std::vector<float> logits(vocab_size, 0.0f);
    logits[0] = 1000.0f;
    logits[1] = 5.0f;
    logits[2] = 4.0f;
    logits[3] = 3.0f;

    // All tokens allowed EXCEPT token 0
    std::vector<int32_t> mask(4, 0xFFFFFFFF);
    mask[0] &= ~(1u << 0);                  // clear bit 0

    SamplingParams params;
    params.temperature = 0.0f;              // greedy
    std::mt19937 rng(42);

    int tok = sample(logits.data(), vocab_size, params, rng, mask.data());

    // Must NOT be 0 — that token is blocked
    if (tok == 0) return false;
    // Must be the best remaining token (1 has logit 5.0)
    return tok == 1;
}

// ---------------------------------------------------------------------------
// TEST 6: sample_unconstrained_no_regression
// ---------------------------------------------------------------------------
/// vocab=1000, token 500 has value 100.0.  Null mask, greedy →
/// must always return 500.
bool sample_unconstrained_no_regression() {
    int vocab_size = 1000;
    std::vector<float> logits(vocab_size, 0.0f);
    logits[500] = 100.0f;

    SamplingParams params;
    params.temperature = 0.0f;              // greedy
    std::mt19937 rng(42);

    for (int i = 0; i < 20; ++i) {
        int tok = sample(logits.data(), vocab_size, params, rng, nullptr);
        if (tok != 500) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// TEST 7: mask_vocab_boundary
// ---------------------------------------------------------------------------
/// vocab_size = 65 → 3 int32 words (bits 0–31, 32–63, and bit 64 alone).
/// Allow ONLY token 64 and verify the partial last word works.
bool mask_vocab_boundary() {
    int vocab_size = 65;
    std::vector<float> logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i)
        logits[i] = static_cast<float>(i);

    // Only token 64 (word 2, bit 0) is allowed
    std::vector<int32_t> mask(3, 0);
    mask[2] = (1u << 0);                    // word=64/32=2, bit=64%32=0

    apply_grammar_mask(logits.data(), vocab_size, mask.data());

    // Token 64 must survive
    if (is_neg_inf(logits[64])) return false;
    if (logits[64] != 64.0f) return false;

    // Tokens 0–63 must be masked
    for (int i = 0; i < 64; ++i)
        if (!is_neg_inf(logits[i])) return false;

    return true;
}

// ---------------------------------------------------------------------------
// TEST 8: mask_nullptr_noop
// ---------------------------------------------------------------------------
/// nullptr mask → logits must be completely unchanged.
bool mask_nullptr_noop() {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int vocab_size = 5;

    std::vector<float> original = logits;

    apply_grammar_mask(logits.data(), vocab_size, nullptr);

    for (int i = 0; i < vocab_size; ++i)
        if (logits[i] != original[i]) return false;

    return true;
}

// ===========================================================================
//  main
// ===========================================================================

int main() {
    struct Test { const char* name; bool (*fn)(); };
    Test tests[] = {
        {"mask_apply_basic",               mask_apply_basic},
        {"mask_apply_all_allowed",         mask_apply_all_allowed},
        {"mask_apply_none_allowed",        mask_apply_none_allowed},
        {"sample_with_mask_deterministic", sample_with_mask_deterministic},
        {"sample_with_mask_excludes_token",sample_with_mask_excludes_token},
        {"sample_unconstrained_no_regression", sample_unconstrained_no_regression},
        {"mask_vocab_boundary",            mask_vocab_boundary},
        {"mask_nullptr_noop",              mask_nullptr_noop},
    };

    int passed = 0, failed = 0;
    for (auto& t : tests) {
        printf("  %-40s ", t.name);
        try {
            if (t.fn()) { printf("PASS\n"); passed++; }
            else        { printf("FAIL\n"); failed++; }
        } catch (const std::exception& e) {
            printf("CRASH: %s\n", e.what());
            failed++;
        }
    }

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
