/// RoPE CUDA kernel — Neox-style, one thread block per (token, head).
/// Also includes a GPU cos/sin kernel to eliminate CPU round-trips.

#include "lightllm/ops/rope.h"
#include <cstdio>
#include <cuda_runtime.h>
#include "lightllm/ops/dispatch.h"

namespace lightllm { namespace ops {
// GPU kernel to compute RoPE cos/sin directly on device
__global__ void rope_cos_sin_kernel(float* cos, float* sin, int num_tokens, int half_dim, float rope_theta){
    int t=blockIdx.x, d=threadIdx.x;
    if(t<num_tokens&&d<half_dim){
        float theta=(float)t/powf(rope_theta,2.f*d/(half_dim*2));
        cos[t*half_dim+d]=cosf(theta);
        sin[t*half_dim+d]=sinf(theta);
    }
}
std::pair<Tensor,Tensor> make_rope_cos_sin(int T,int hd,float theta){
    Tensor cos({T,hd/2},DType::F32,Device::CUDA),sin({T,hd/2},DType::F32,Device::CUDA);
    rope_cos_sin_kernel<<<T,std::min(256,hd/2)>>>(cos.data<float>(),sin.data<float>(),T,hd/2,theta);
    return {std::move(cos),std::move(sin)};
}
} }

namespace lightllm {
namespace ops {

template <typename T>
__global__ void rope_kernel(
    T* __restrict__ x,           // in-place on Q or K
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    int num_tokens,
    int num_heads,
    int head_dim,
    int rotary_dim,
    int stride_token,
    int stride_head) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = num_tokens * num_heads;
    if (tid >= total_threads) return;

    int token_idx = tid / num_heads;
    int head_idx = tid % num_heads;
    int half_rotary = rotary_dim / 2;

    T* row = x + token_idx * stride_token + head_idx * stride_head;
    const float* c = cos + token_idx * half_rotary;
    const float* s = sin + token_idx * half_rotary;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x) {
        float x1 = static_cast<float>(row[i]);
        float x2 = static_cast<float>(row[i + half_rotary]);
        float o1 = x1 * c[i] - x2 * s[i];
        float o2 = x2 * c[i] + x1 * s[i];
        row[i] = static_cast<T>(o1);
        row[i + half_rotary] = static_cast<T>(o2);
    }
}

void rope(Tensor& q, Tensor* k, const Tensor& cos, const Tensor& sin) {
    int num_tokens = q.size(0);
    int num_heads = q.size(1);
    int head_dim = q.size(2);
    int rotary_dim = cos.size(1) * 2;
    int half_rotary = rotary_dim / 2;
    int block = std::min(256, std::max(half_rotary, 1));
    int grid = num_tokens * num_heads;

    DISPATCH_FLOAT_HALF(q.dtype(), "rope", {
        rope_kernel<scalar_t>
            <<<grid, block>>>(
                q.data<scalar_t>(),
                cos.data<float>(), sin.data<float>(),
                num_tokens, num_heads, head_dim, rotary_dim,
                q.size(1) * q.size(2),  // stride_token
                q.size(2));              // stride_head
    });
    
    if (k) {
        int num_kv_heads = k->size(1);
        DISPATCH_FLOAT_HALF(k->dtype(), "rope_k", {
            rope_kernel<scalar_t>
                <<<grid, block>>>(
                    k->data<scalar_t>(),
                    cos.data<float>(), sin.data<float>(),
                    num_tokens, num_kv_heads, head_dim, rotary_dim,
                    k->size(1) * k->size(2),
                    k->size(2));
        });
    }
}

}  // namespace ops
}  // namespace lightllm
