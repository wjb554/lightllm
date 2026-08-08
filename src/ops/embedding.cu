/// Embedding lookup — one thread per output element.
#include "lightllm/ops/embedding.h"
#include <cuda_runtime.h>

namespace lightllm {
namespace ops {

template<typename T>
__global__ void embedding_kernel(T* out, const int* ids, const T* weight,
                                  int num_tokens, int hidden_size, int vocab_size) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int row=tid/hidden_size, col=tid%hidden_size;
    if(row>=num_tokens)return;
    int token_id=ids[row];
    if(token_id<0||token_id>=vocab_size)token_id=0; // safety clamp
    out[tid]=weight[token_id*hidden_size+col];
}

Tensor embedding(const Tensor& token_ids, const Tensor& weight) {
    int num_tokens=token_ids.numel();
    int hidden_size=weight.size(1);
    Tensor out({num_tokens,hidden_size},weight.dtype(),Device::CUDA);
    int N=num_tokens*hidden_size, blk=256;
    int grid=(N+blk-1)/blk;
    // fp32 or bf16 — use raw pointer dispatch
    float*op=(float*)out.raw();const float*wp=(const float*)weight.raw();
    embedding_kernel<float><<<grid,blk>>>(op,(const int*)token_ids.raw(),wp,
        num_tokens,hidden_size,weight.size(0));
    return std::move(out);
}

}  // namespace ops
}  // namespace lightllm
