/// Softmax — 3-kernel GPU with proper cross-warp reduction.
#include "lightllm/ops/softmax.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace lightllm {
namespace ops {

template<typename T>
__global__ void sm_max_kernel(float* row_buf, const T* x, int cols){
    int row=blockIdx.x, tid=threadIdx.x;
    __shared__ float warp_max[32]; // up to 32 warps
    int wid=tid/32, lid=tid%32;
    float mx=-1e38f;
    for(int i=tid;i<cols;i+=blockDim.x)mx=fmaxf(mx,float(x[row*cols+i]));
    for(int o=16;o>0;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));
    if(lid==0)warp_max[wid]=mx;
    __syncthreads();
    // Thread 0 reduces across warps
    if(tid==0){
        int nw=blockDim.x/32;
        for(int w=1;w<nw;w++)mx=fmaxf(mx,warp_max[w]);
        row_buf[row]=mx;
    }
}

template<typename T>
__global__ void sm_apply_kernel(T* x, float* row_buf, int cols){
    int row=blockIdx.x, tid=threadIdx.x;
    __shared__ float warp_sum[32];
    int wid=tid/32, lid=tid%32;
    float mx=row_buf[row], sm=0;
    for(int i=tid;i<cols;i+=blockDim.x){float v=expf(float(x[row*cols+i])-mx);x[row*cols+i]=T(v);sm+=v;}
    for(int o=16;o>0;o>>=1)sm+=__shfl_down_sync(0xffffffff,sm,o);
    if(lid==0)warp_sum[wid]=sm;
    __syncthreads();
    if(tid==0){
        int nw=blockDim.x/32;
        for(int w=1;w<nw;w++)sm+=warp_sum[w];
        row_buf[row]=sm;
    }
}

template<typename T>
__global__ void sm_norm_kernel(T* x, const float* row_buf, int cols){
    int row=blockIdx.x, tid=threadIdx.x;
    float inv=1.f/(row_buf[row]+1e-8f);
    for(int i=tid;i<cols;i+=blockDim.x)x[row*cols+i]=T(float(x[row*cols+i])*inv);
}

Tensor softmax(const Tensor& x){
    int rows=x.numel()/x.size(-1), cols=x.size(-1);
    int blk=(cols<256)?(1<<((int)ceilf(log2f((float)cols)))):256;if(blk<1)blk=32;
    float *d_buf=nullptr,*d_data=nullptr;
    cudaMalloc(&d_buf,rows*sizeof(float));
    cudaMalloc(&d_data,rows*cols*sizeof(float));
    cudaMemcpy(d_data,x.raw(),rows*cols*sizeof(float),cudaMemcpyDeviceToDevice);
    if(x.dtype()==DType::F32){
        sm_max_kernel<float><<<rows,blk>>>(d_buf,(const float*)d_data,cols);
        sm_apply_kernel<float><<<rows,blk>>>(d_data,d_buf,cols);
        sm_norm_kernel<float><<<rows,blk>>>(d_data,d_buf,cols);
    }else{cudaFree(d_buf);cudaFree(d_data);throw std::runtime_error("softmax: fp32 only");}
    Tensor out(x.shape(),x.dtype(),Device::CUDA);
    cudaMemcpy(out.raw(),d_data,rows*cols*sizeof(float),cudaMemcpyDeviceToDevice);
    cudaFree(d_buf);cudaFree(d_data);
    return std::move(out);
}

}  // namespace ops
}  // namespace lightllm
