/// LightLLM Sampling — CPU-based token selection from logits.
///
/// Implements temperature scaling, top-k, top-p, min-p filtering, and
/// multinomial sampling.  Entire pipeline runs on CPU (see sampling.h
/// for design rationale).
#include "lightllm/ops/sampling.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lightllm {
namespace ops {

// ===========================================================================
//  sample — draw a single token from raw logits
// ===========================================================================

int sample(const float* logits, int vocab_size,
           const SamplingParams& params, std::mt19937& rng,
           const int32_t* grammar_mask) {

    // ------------------------------------------------------------------
    // 0.  Defensive guards
    // ------------------------------------------------------------------
    if (vocab_size <= 0) return 0;

    // ------------------------------------------------------------------
    // 1.  Greedy / near-deterministic short-circuit
    // ------------------------------------------------------------------
    //  temperature < 1e-5  behaves indistinguishably from greedy but
    //  avoids division by a tiny number in step 2.
    if (params.temperature < 1e-5f) {
        // Apply grammar mask even for greedy — search best *allowed* token.
        int best = -1;
        float best_val = -INFINITY;
        for (int i = 0; i < vocab_size; ++i) {
            if (grammar_mask != nullptr) {
                int word = i / 32;
                int bit  = i % 32;
                if (!(grammar_mask[word] & (1u << bit))) continue;
            }
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }
        // Fallback: if mask blocked everything, return global argmax
        if (best < 0) {
            best = 0;
            best_val = logits[0];
            for (int i = 1; i < vocab_size; ++i)
                if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        }
        return best;
    }

    // ------------------------------------------------------------------
    // 3.  Copy logits into a mutable working buffer
    // ------------------------------------------------------------------
    std::vector<float> work(logits, logits + vocab_size);
    if (grammar_mask != nullptr) {
        apply_grammar_mask(work.data(), vocab_size, grammar_mask);
        // Verify at least one token remains
        bool any = false;
        for (int i = 0; i < vocab_size; ++i) if (work[i] > -INFINITY) { any = true; break; }
        if (!any) {
            int best = 0; float best_val = logits[0];
            for (int i = 1; i < vocab_size; ++i) if (logits[i] > best_val) { best_val = logits[i]; best = i; }
            return best;
        }
    }

    // ------------------------------------------------------------------
    // 4.  Temperature scaling
    // ------------------------------------------------------------------
    //  Guard against division by zero with max(temp, 1e-8).
    const float inv_temp = 1.0f / std::max(params.temperature, 1e-8f);
    for (int i = 0; i < vocab_size; ++i) {
        work[i] *= inv_temp;
    }

    // ------------------------------------------------------------------
    // 5.  Top-k filtering (on logits, before softmax)
    // ------------------------------------------------------------------
    //  Requirement: create (value, index) pairs, partial_sort the top-k,
    //  set all other logits to -inf.
    if (params.top_k > 0 && params.top_k < vocab_size) {
        const int k = params.top_k;

        // Build (value, index) pairs for non-masked candidates
        std::vector<std::pair<float, int>> pairs;
        pairs.reserve(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            if (work[i] > -INFINITY) {
                pairs.emplace_back(work[i], i);
            }
        }

        const int n_candidates = static_cast<int>(pairs.size());
        if (n_candidates > k) {
            // partial_sort the top-k (descending)
            std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                [](const std::pair<float, int>& a,
                   const std::pair<float, int>& b) {
                    return a.first > b.first;
                });

            const float threshold = pairs[k - 1].first;

            // Mask everything below the k-th largest value
            int kept = 0;
            for (int i = 0; i < vocab_size; ++i) {
                if (work[i] < threshold) {
                    work[i] = -INFINITY;
                } else {
                    ++kept;
                }
            }

            // Safety: if filtering zeroed everything, fallback to argmax
            if (kept == 0) {
                int best = 0;
                float best_val = logits[0];
                for (int i = 1; i < vocab_size; ++i) {
                    if (logits[i] > best_val) {
                        best_val = logits[i];
                        best = i;
                    }
                }
                return best;
            }
        }
    }

    // ------------------------------------------------------------------
    // 6.  Numerically stable softmax
    // ------------------------------------------------------------------
    //  Standard three-pass softmax: find max, exp-and-accumulate, scale.
    {
        // Pass 1 — find max (ignoring -inf entries)
        float max_val = -INFINITY;
        for (int i = 0; i < vocab_size; ++i) {
            if (work[i] > max_val) max_val = work[i];
        }

        // If every logit was -inf (impossible after the guards above, but
        // handle gracefully), fallback to argmax on the original logits.
        if (max_val == -INFINITY) {
            int best = 0;
            float best_val = logits[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best = i;
                }
            }
            return best;
        }

        // Pass 2 — exp and accumulate
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            if (work[i] > -INFINITY) {
                work[i] = std::exp(work[i] - max_val);
                sum += work[i];
            } else {
                work[i] = 0.0f;
            }
        }

        // Pass 3 — normalize (guard against zero sum)
        if (sum > 0.0f) {
            const float inv_sum = 1.0f / sum;
            for (int i = 0; i < vocab_size; ++i) {
                work[i] *= inv_sum;
            }
        } else {
            // All exp'd values underflowed — fallback to argmax
            int best = 0;
            float best_val = logits[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best = i;
                }
            }
            return best;
        }
    }

    // ------------------------------------------------------------------
    // 7.  Top-p (nucleus) filtering on softmax probabilities
    // ------------------------------------------------------------------
    if (params.top_p < 1.0f) {
        // Sort indices by descending probability
        std::vector<int> indices(vocab_size);
        for (int i = 0; i < vocab_size; ++i) indices[i] = i;

        std::sort(indices.begin(), indices.end(),
            [&work](int a, int b) { return work[a] > work[b]; });

        // Build cumulative softmax sum and find cutoff
        float cum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; ++i) {
            cum += work[indices[i]];
            if (cum >= params.top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out tokens beyond the nucleus
        for (int i = cutoff; i < vocab_size; ++i) {
            work[indices[i]] = 0.0f;
        }

        // Renormalize the kept set
        {
            float sum = 0.0f;
            for (int i = 0; i < vocab_size; ++i) sum += work[i];

            if (sum > 0.0f) {
                const float inv_sum = 1.0f / sum;
                for (int i = 0; i < vocab_size; ++i) work[i] *= inv_sum;
            } else {
                // Top-p filtered everything — fallback to argmax
                int best = 0;
                float best_val = logits[0];
                for (int i = 1; i < vocab_size; ++i) {
                    if (logits[i] > best_val) {
                        best_val = logits[i];
                        best = i;
                    }
                }
                return best;
            }
        }
    }

    // ------------------------------------------------------------------
    // 8.  Min-p filtering on softmax probabilities
    // ------------------------------------------------------------------
    if (params.min_p > 0.0f) {
        const float max_p = *std::max_element(work.data(),
                                               work.data() + vocab_size);
        const float threshold = max_p * params.min_p;

        for (int i = 0; i < vocab_size; ++i) {
            if (work[i] < threshold) work[i] = 0.0f;
        }

        // Renormalize
        {
            float sum = 0.0f;
            for (int i = 0; i < vocab_size; ++i) sum += work[i];

            if (sum > 0.0f) {
                const float inv_sum = 1.0f / sum;
                for (int i = 0; i < vocab_size; ++i) work[i] *= inv_sum;
            } else {
                // Min-p filtered everything — fallback to argmax
                int best = 0;
                float best_val = logits[0];
                for (int i = 1; i < vocab_size; ++i) {
                    if (logits[i] > best_val) {
                        best_val = logits[i];
                        best = i;
                    }
                }
                return best;
            }
        }
    }

    // ------------------------------------------------------------------
    // 9.  Multinomial draw via binary search on the CDF
    // ------------------------------------------------------------------
    //  Build the cumulative distribution function in-place.
    for (int i = 1; i < vocab_size; ++i) {
        work[i] += work[i - 1];
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    // Binary search: find the first token whose CDF >= r.
    //  std::lower_bound returns the first iterator where *it >= r.
    int idx = static_cast<int>(
        std::lower_bound(work.data(), work.data() + vocab_size, r) -
        work.data());

    // Floating-point edge case:  r may be just above CDF[-1] (~1.0) due
    // to rounding, causing lower_bound to return end().  Walk backward
    // to find the last token with a positive probability mass.
    if (idx >= vocab_size) {
        for (int i = vocab_size - 1; i >= 0; --i) {
            if (i == 0 || work[i] > work[i - 1]) return i;
        }
        return 0;
    }

    // Guard against r == 0 landing on a zero-probability tail.
    // When prob[i]==0 the CDF stays flat across i.  Scan forward until
    // the first token with mass > 0 (CDF value increases).
    {
        int start_idx = idx;
        while (idx < vocab_size) {
            // For idx==0, probability mass exists iff work[0] > 0
            // and differs from work[1] (work[0]==work[1] => prob[0]==0).
            // For idx>0, work[idx]==work[idx-1] means prob[idx]==0.
            bool has_mass = true;
            if (idx == 0) {
                has_mass = (work[0] > 0.0f);
            } else {
                has_mass = (work[idx] != work[idx - 1]);
            }
            if (has_mass) break;
            ++idx;
        }
        if (idx >= vocab_size) {
            // Walk backward to last token with mass
            for (int i = vocab_size - 1; i >= 0; --i) {
                if (i == 0 || work[i] > work[i - 1]) return i;
            }
            return 0;
        }
    }

    return idx;
}

// ===========================================================================
//  sample_batch — sample a batch of tokens (one per sequence)
// ===========================================================================

void sample_batch(int* out_tokens, const float* logits,
                  int batch_size, int vocab_size,
                  const SamplingParams& params, uint64_t seed) {

    // Resolve base seed
    if (seed == 0) {
        std::random_device rd;
        seed = static_cast<uint64_t>(rd()) |
               (static_cast<uint64_t>(rd()) << 32);
    }

    // Each sequence gets its own RNG state derived from base_seed + index,
    // guaranteeing independent and reproducible draws.
    for (int i = 0; i < batch_size; ++i) {
        std::mt19937 rng(static_cast<unsigned>(seed + static_cast<uint64_t>(i)));
        out_tokens[i] = sample(logits + i * vocab_size, vocab_size,
                                params, rng);
    }
}

void apply_grammar_mask(float* logits, int vocab_size, const int32_t* mask) {
    if (!mask) return;
    for (int i = 0; i < vocab_size; ++i) {
        int word = i / 32;
        int bit  = i % 32;
        if (!(mask[word] & (1u << bit))) {
            logits[i] = -INFINITY;
        }
    }
}

}  // namespace ops
}  // namespace lightllm
