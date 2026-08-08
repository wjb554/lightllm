/// Activation functions — one kernel per function, one thread per element.
#include "lightllm/ops/activation.h"
#include "lightllm/ops/dispatch.h"

namespace lightllm {
namespace ops {

template<typename T>
__global__ void relu_kernel(T* x, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) x[i]=T(fmaxf(0.f,float(x[i])));
}
void relu(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"relu",{
    relu_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

template<typename T>
__global__ void sigmoid_kernel(T* x, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){float v=float(x[i]);x[i]=T(1.f/(1.f+expf(-v)));}
}
void sigmoid(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"sigmoid",{
    sigmoid_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

template<typename T>
__global__ void tanh_kernel(T* x, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){float v=float(x[i]);x[i]=T(tanhf(v));}
}
void tanh_op(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"tanh",{
    tanh_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

template<typename T>
__global__ void gelu_kernel(T* x, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){float v=float(x[i]);x[i]=T(v*0.5f*(1.f+tanhf(0.79788456f*(v+0.044715f*v*v*v))));}
}
void gelu(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"gelu",{
    gelu_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

// Swish = x * sigmoid(x) = SiLU (same as our elementwise silu)
template<typename T>
__global__ void swish_kernel(T* x, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){float v=float(x[i]);x[i]=T(v/(1.f+expf(-v)));}
}
void swish(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"swish",{
    swish_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

template<typename T>
__global__ void mish_kernel(T* x, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){float v=float(x[i]);x[i]=T(v*tanhf(logf(1.f+expf(v))));}
}
void mish(Tensor& x){int N=int(x.numel());DISPATCH_FLOAT_TYPES(x.dtype(),"mish",{
    mish_kernel<scalar_t><<<(N+255)/256,256>>>(x.data<scalar_t>(),N);});}

}  // namespace ops
}  // namespace lightllm
