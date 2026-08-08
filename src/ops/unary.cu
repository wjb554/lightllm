/// Element-wise unary ops — neg, abs, sign, clip, round/floor/ceil, pow, sqrt, exp, log
#include "lightllm/ops/unary.h"
#include "lightllm/ops/dispatch.h"

namespace lightllm {
namespace ops {

template<typename T> __global__ void neg_kernel(T* x,int N){int i=blockIdx.x*256+threadIdx.x;if(i<N)x[i]=T(-float(x[i]));}
void neg(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"neg",{
    neg_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

template<typename T> __global__ void abs_kernel(T* x,int N){int i=blockIdx.x*256+threadIdx.x;if(i<N)x[i]=T(fabsf(float(x[i])));}
void abs_op(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"abs",{
    abs_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

template<typename T> __global__ void sign_kernel(T* x,int N){int i=blockIdx.x*256+threadIdx.x;if(i<N){float v=float(x[i]);x[i]=T(v>0?1.f:(v<0?-1.f:0.f));}}
void sign(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"sign",{
    sign_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

template<typename T> __global__ void clip_kernel(T* x,float lo,float hi,int N){int i=blockIdx.x*256+threadIdx.x;if(i<N){float v=float(x[i]);x[i]=T(fminf(hi,fmaxf(lo,v)));}}
void clip(Tensor& x,float lo,float hi){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"clip",{
    clip_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),lo,hi,N);});}

template<typename T> __global__ void round_kernel(T* x,int N){int i=blockIdx.x*256+threadIdx.x;if(i<N)x[i]=T(roundf(float(x[i])));}
void round_op(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"round",{
    round_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

template<typename T> __global__ void floor_kernel(T* x,int N){int i=blockIdx.x*256+threadIdx.x;if(i<N)x[i]=T(floorf(float(x[i])));}
void floor_op(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"floor",{
    floor_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

template<typename T> __global__ void ceil_kernel(T* x,int N){int i=blockIdx.x*256+threadIdx.x;if(i<N)x[i]=T(ceilf(float(x[i])));}
void ceil_op(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"ceil",{
    ceil_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

template<typename T> __global__ void pow_kernel(T* x,float e,int N){int i=blockIdx.x*256+threadIdx.x;if(i<N)x[i]=T(powf(float(x[i]),e));}
void pow_op(Tensor& x,float exp){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"pow",{
    pow_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),exp,N);});}

template<typename T> __global__ void sqrt_kernel(T* x,int N){int i=blockIdx.x*256+threadIdx.x;if(i<N)x[i]=T(sqrtf(float(x[i])));}
void sqrt_op(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"sqrt",{
    sqrt_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

template<typename T> __global__ void exp_kernel(T* x,int N){int i=blockIdx.x*256+threadIdx.x;if(i<N)x[i]=T(expf(float(x[i])));}
void exp_op(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"exp",{
    exp_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

template<typename T> __global__ void log_kernel(T* x,int N){int i=blockIdx.x*256+threadIdx.x;if(i<N)x[i]=T(logf(float(x[i])));}
void log_op(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"log",{
    log_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

}  // namespace ops
}  // namespace lightllm
