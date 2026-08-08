#pragma once
/// LightLLM Sampling — temperature, top-k, top-p, min-p, multinomial.
///
/// All sampling runs on CPU because:
///  - The lm_head logits (151k floats per token) are already computed on CPU
///    in the decode loop (see engine.cpp lines 209-212).
///  - Copying 600 KB to GPU for a 1-token decode is pointless when the
///    GEMM that produces those logits takes 100x longer.
///  - This matches llama.cpp and vLLM practice.
///
/// Algorithm (order matters — each step sees the previous step's output):
///
///   1.  temperature == 0  ->  greedy argmax (short-circuit, no allocations)
///   2.  temperature scaling:  logits[i] /= temperature
///   3.  top-k  filtering:  if top_k > 0, find k-th largest value via
///       std::nth_element (O(n)), mask everything below to -inf
///   4.  top-p  filtering:  sort by descending probability after an
///       intermediate softmax, accumulate CDF, truncate at top_p
///   5.  min-p  filtering:  after the final softmax, zero out any token
///       whose probability is below min_p * max_prob, then renormalize
///   6.  Final softmax -> multinomial draw via CDF + uniform(0,1)
///
///   Step ordering rationale:  top-k trims the long tail cheaply (O(n)
///   selection vs O(n log n) full sort).  top-p comes before min-p so
///   the nucleus already removed the bulk of the noise.  min-p runs last
///   because it depends on the softmax-normalised max probability.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace lightllm {
namespace ops {

// ---------------------------------------------------------------------------
// SamplingParams — per-request parameters controlling token selection
// ---------------------------------------------------------------------------
struct SamplingParams {
    /// Temperature for logit scaling.  Higher = more random.
    ///   0.0  ->  greedy argmax (deterministic).
    ///   1.0  ->  no scaling (default).
    ///   >1.0 ->  flattens distribution (more diverse).
    float temperature = 1.0f;

    /// Top-k sampling: keep only the k most probable tokens.
    ///   0   ->  disabled (keep all tokens).
    ///   >0  ->  all logits below the k-th largest are set to -inf.
    int top_k = 0;

    /// Top-p (nucleus) sampling: keep the smallest set of tokens whose
    /// cumulative probability is >= top_p.
    ///   1.0  ->  disabled (keep all tokens).
    ///   0.95 ->  typical nucleus setting.
    float top_p = 1.0f;

    /// Min-p sampling: after softmax, discard tokens whose probability
    /// is less than min_p * max_probability.
    ///   0.0  ->  disabled.
    ///   0.05 ->  typical filter (removes tokens < 5% of the top token).
    float min_p = 0.0f;

    /// Seed for the per-call random generator.  When batching, the
    /// caller adds the sample index to this seed so every sequence
    /// in the batch gets a statistically independent draw.
    ///   seed == 0  ->  seed from std::random_device{} (non-reproducible).
    uint64_t seed = 42;
};

// ---------------------------------------------------------------------------
// sample — draw a single token from logits
// ---------------------------------------------------------------------------
/// Sample one token given unnormalised logits and sampling parameters.
///
/// \param logits      Pointer to `vocab_size` floats (raw logits / scores).
/// \param vocab_size  Number of tokens in the vocabulary.
/// \param params      Sampling parameters (see SamplingParams).
/// \param rng         Mersenne Twister instance, seeded by the caller.
///                    For batch mode advance the same generator or pass
///                    a per-sequence generator seeded with `seed + seq_idx`.
/// \return            The sampled token id in [0, vocab_size).
///
/// Complexity:  O(vocab_size) when greedy;
///              O(vocab_size + k) when top_k only;
///              O(vocab_size log vocab_size) when top_p is active
///              (dominant cost is sorting 151k floats ~ 2.6M compares).
///
/// Thread safety:  not reentrant — only `rng` is mutated.  Multiple
/// callers must each own their `std::mt19937`.
int sample(const float* logits, int vocab_size,
           const SamplingParams& params, std::mt19937& rng);

// ---------------------------------------------------------------------------
// sample_batch — sample a batch of tokens (one per sequence)
// ---------------------------------------------------------------------------
/// Sample `batch_size` tokens in parallel on CPU.
///
/// \param out_tokens  [batch_size] output token ids.
/// \param logits      [batch_size * vocab_size] row-major logits.
/// \param batch_size  Number of independent sequences.
/// \param vocab_size  Vocabulary size (same for all sequences).
/// \param params      Shared sampling parameters (same for every sequence).
/// \param seed        Base seed — sequence i uses seed + i.
///
/// Each row is sampled independently.  When params.seed == 0 the base
/// seed is drawn from std::random_device{} once.
void sample_batch(int* out_tokens, const float* logits,
                  int batch_size, int vocab_size,
                  const SamplingParams& params, uint64_t seed);

// ===========================================================================
//  Inline helpers (used internally by sample / sample_batch)
// ===========================================================================

namespace detail {

/// Numerically stable softmax in-place.  Returns the sum (will be > 0 if
/// at least one non -inf logit exists).
inline float softmax_inplace(float* x, int n) {
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < n; ++i)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < n; ++i) x[i] *= inv;
    }
    return sum;
}

/// Multinomial draw from a probability vector (must sum to 1).
/// Returns the sampled index, or 0 if all probabilities are zero.
inline int multinomial_draw(const float* probs, int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (int i = 0; i < n; ++i) {
        cum += probs[i];
        if (r <= cum) return i;
    }
    // Floating-point edge case: fall back to last non-zero token
    for (int i = n - 1; i >= 0; --i)
        if (probs[i] > 0.0f) return i;
    return 0;
}

/// Apply top-k filter in-place: set all but the k largest values to -inf.
/// Returns the number of non-masked tokens (always positive when k > 0).
inline int apply_top_k(float* logits, int n, int k) {
    if (k <= 0 || k >= n) return n;

    // std::nth_element is O(n) — far cheaper than full sort for large n
    std::vector<float> copy(logits, logits + n);
    std::nth_element(copy.begin(), copy.begin() + k - 1, copy.end(),
                     std::greater<float>());
    float threshold = copy[k - 1];

    int kept = 0;
    for (int i = 0; i < n; ++i) {
        if (logits[i] < threshold) logits[i] = -INFINITY;
        else ++kept;
    }
    return kept;
}

/// Apply min-p filter in-place on *softmax-normalised* probabilities.
/// Discards tokens with prob < min_p * max_prob, then renormalizes.
inline void apply_min_p(float* probs, int n, float min_p) {
    if (min_p <= 0.0f) return;

    float max_p = *std::max_element(probs, probs + n);
    float threshold = max_p * min_p;

    for (int i = 0; i < n; ++i)
        if (probs[i] < threshold) probs[i] = 0.0f;

    // Renormalize
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += probs[i];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < n; ++i) probs[i] *= inv;
    }
}

/// Apply top-p filter in-place on *softmax-normalised* probabilities.
/// Keeps the smallest suffix of tokens (when sorted descending) whose
/// cumulative probability >= top_p.  Renormalizes afterward.
inline void apply_top_p(float* probs, int n, float top_p) {
    if (top_p >= 1.0f) return;

    // Build sorted index permutation (descending probability order).
    // With vocab_size = 151936 this sort costs ~2.6 M comparisons — still
    // negligible next to the lm_head GEMM.
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return probs[a] > probs[b];
    });

    float cum = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; ++i) {
        cum += probs[idx[i]];
        if (cum >= top_p) { cutoff = i + 1; break; }
    }

    for (int i = cutoff; i < n; ++i)
        probs[idx[i]] = 0.0f;

    // Renormalize
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += probs[i];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < n; ++i) probs[i] *= inv;
    }
}

}  // namespace detail

}  // namespace ops
}  // namespace lightllm
