/// Naive attention — CPU-side loops for correctness (Phase 2 baseline).
#include "lightllm/ops/attention.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include "lightllm/ops/dispatch.h"

namespace lightllm {
namespace ops {

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v) {
    int T = q.size(0), Hq = q.size(1), Hkv = k.size(1), D = q.size(2);
    if (Hq % Hkv != 0) throw std::runtime_error("attention: GQA ratio must be integer");
    int groups = Hq / Hkv;
    float scale = 1.0f / sqrtf(float(D));

    // CPU reference: compute Q@K^T, softmax, S@V entirely on host for now.
    // Phase 3 will move these to CUDA kernels.
    auto q_cpu = std::vector<float>(T * Hq * D);
    auto k_cpu = std::vector<float>(T * Hkv * D);
    auto v_cpu = std::vector<float>(T * Hkv * D);
    q.copy_to(q_cpu.data(), q.nbytes());
    k.copy_to(k_cpu.data(), k.nbytes());
    v.copy_to(v_cpu.data(), v.nbytes());

    // Q @ K^T -> scores [T, Hq, T]
    std::vector<float> scores(T * Hq * T, 0.0f);
    for (int h = 0; h < Hq; h++) {
        int kvh = h / groups;
        for (int t = 0; t < T; t++) {
            for (int s = 0; s < T; s++) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += q_cpu[(t * Hq + h) * D + d] *
                           k_cpu[(s * Hkv + kvh) * D + d];
                scores[(t * Hq + h) * T + s] = dot * scale;
            }
        }
    }

    // Softmax per row
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < Hq; h++) {
            float max_val = -INFINITY;
            for (int s = 0; s < T; s++)
                max_val = fmaxf(max_val, scores[(t * Hq + h) * T + s]);
            float sum = 0.0f;
            for (int s = 0; s < T; s++) {
                float v = expf(scores[(t * Hq + h) * T + s] - max_val);
                scores[(t * Hq + h) * T + s] = v;
                sum += v;
            }
            for (int s = 0; s < T; s++)
                scores[(t * Hq + h) * T + s] /= sum;
        }
    }

    // S @ V -> output [T, Hq, D]
    std::vector<float> out_cpu(T * Hq * D, 0.0f);
    for (int h = 0; h < Hq; h++) {
        int kvh = h / groups;
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int s = 0; s < T; s++)
                    acc += scores[(t * Hq + h) * T + s] *
                           v_cpu[(s * Hkv + kvh) * D + d];
                out_cpu[(t * Hq + h) * D + d] = acc;
            }
        }
    }

    Tensor out({T, Hq, D}, q.dtype(), Device::CUDA);
    out.copy_from(out_cpu.data(), out.nbytes());
    return std::move(out);
}

}  // namespace ops
}  // namespace lightllm
