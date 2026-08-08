/// Reduction ops — warp-level reduce, top-k CPU baseline.
#include "lightllm/ops/reduce.h"
#include "lightllm/ops/dispatch.h"
#include <algorithm>

namespace lightllm {
namespace ops {

template<typename T>
__global__ void sum_kernel(float* out, const T* x, int rows, int cols) {
    int row=blockIdx.x, tid=threadIdx.x; float acc=0;
    for(int i=tid;i<cols;i+=blockDim.x)acc+=float(x[row*cols+i]);
    for(int o=16;o>0;o>>=1)acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(tid==0)out[row]=acc;
}
template<typename T>
__global__ void max_kernel(float* out, const T* x, int rows, int cols) {
    int row=blockIdx.x, tid=threadIdx.x; float acc=-1e38f;
    for(int i=tid;i<cols;i+=blockDim.x)acc=fmaxf(acc,float(x[row*cols+i]));
    for(int o=16;o>0;o>>=1)acc=fmaxf(acc,__shfl_down_sync(0xffffffff,acc,o));
    if(tid==0)out[row]=acc;
}
template<typename T>
__global__ void min_kernel(float* out, const T* x, int rows, int cols) {
    int row=blockIdx.x, tid=threadIdx.x; float acc=1e38f;
    for(int i=tid;i<cols;i+=blockDim.x)acc=fminf(acc,float(x[row*cols+i]));
    for(int o=16;o>0;o>>=1)acc=fminf(acc,__shfl_down_sync(0xffffffff,acc,o));
    if(tid==0)out[row]=acc;
}
template<typename T>
__global__ void scale_kernel(float* x, int N, float inv){int i=blockIdx.x*256+threadIdx.x;if(i<N)x[i]*=inv;}

Tensor reduce_sum(const Tensor& x){
    int rows=x.numel()/x.size(-1),cols=x.size(-1);
    Tensor out({rows},DType::F32,Device::CUDA);
    int blk=std::min(256,next_pow2(cols));
    DISPATCH_FLOAT_TYPES(x.dtype(),"sum",
        {sum_kernel<scalar_t><<<rows,blk>>>(out.data<float>(),x.data<scalar_t>(),rows,cols);});
    return std::move(out);
}
Tensor reduce_max(const Tensor& x){
    int rows=x.numel()/x.size(-1),cols=x.size(-1);
    Tensor out({rows},DType::F32,Device::CUDA);
    int blk=std::min(256,next_pow2(cols));
    DISPATCH_FLOAT_TYPES(x.dtype(),"max",
        {max_kernel<scalar_t><<<rows,blk>>>(out.data<float>(),x.data<scalar_t>(),rows,cols);});
    return std::move(out);
}
Tensor reduce_min(const Tensor& x){
    int rows=x.numel()/x.size(-1),cols=x.size(-1);
    Tensor out({rows},DType::F32,Device::CUDA);
    int blk=std::min(256,next_pow2(cols));
    DISPATCH_FLOAT_TYPES(x.dtype(),"min",
        {min_kernel<scalar_t><<<rows,blk>>>(out.data<float>(),x.data<scalar_t>(),rows,cols);});
    return std::move(out);
}
Tensor reduce_mean(const Tensor& x){
    auto s=reduce_sum(x);int N=s.numel();
    scale_kernel<float><<<(N+255)/256,256>>>(s.data<float>(),N,1.f/x.size(-1));
    return s;
}
std::pair<Tensor,Tensor> topk(const Tensor& x, int k){
    int rows=x.numel()/x.size(-1),cols=x.size(-1);
    if(k>cols)throw std::runtime_error("topk: k > cols");
    Tensor vals({rows,k},DType::F32,Device::CUDA),idxs({rows,k},DType::I32,Device::CUDA);
    std::vector<float> hx(x.numel()),hv(rows*k);
    std::vector<int32_t> hi(rows*k);
    x.copy_to(hx.data(),x.nbytes());
    for(int r=0;r<rows;r++){
        std::vector<std::pair<float,int>> tmp(cols);
        for(int c=0;c<cols;c++)tmp[c]={hx[r*cols+c],c};
        std::partial_sort(tmp.begin(),tmp.begin()+k,tmp.end(),std::greater<>());
        for(int i=0;i<k;i++){hv[r*k+i]=tmp[i].first;hi[r*k+i]=tmp[i].second;}
    }
    vals.copy_from(hv.data(),vals.nbytes());idxs.copy_from(hi.data(),idxs.nbytes());
    return {std::move(vals),std::move(idxs)};
}

}  // namespace ops
}  // namespace lightllm
