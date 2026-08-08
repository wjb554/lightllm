/// GEMM — progressive fp32 backends + cuBLAS fp16 (Tensor Core)
#include "lightllm/ops/gemm.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace lightllm {
namespace ops {

GemmBackend g_gemm_backend = GemmBackend::CuBLAS;
void set_gemm_backend(GemmBackend b) { g_gemm_backend = b; }
GemmBackend get_gemm_backend() { return g_gemm_backend; }
static cublasHandle_t g_handle = nullptr;

// ====== FP32 Backends ======

template <typename T>
__global__ void gemm_naive_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    int r=blockIdx.y*blockDim.y+threadIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=M||c>=N)return; float acc=0;
    for(int k=0;k<K;k++)acc+=float(A[r*K+k])*float(B[k*N+c]);
    C[r*N+c]=T(acc);
}

template <typename T>
__global__ void gemm_tiled_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    constexpr int TM=16,TN=16,TK=8;
    __shared__ float As[TM][TK],Bs[TK][TN];
    int br=blockIdx.y*TM,bc=blockIdx.x*TN,ty=threadIdx.y,tx=threadIdx.x;float acc=0;
    for(int kb=0;kb<(K+TK-1)/TK;kb++){
        for(int i=ty*TN+tx;i<TM*TK;i+=256){int m=i/TK,k=i%TK;As[m][k]=(br+m<M&&kb*TK+k<K)?float(A[(br+m)*K+kb*TK+k]):0.f;}
        for(int i=ty*TN+tx;i<TK*TN;i+=256){int k=i/TN,n=i%TN;Bs[k][n]=(kb*TK+k<K&&bc+n<N)?float(B[(kb*TK+k)*N+bc+n]):0.f;}
        __syncthreads();
        for(int k=0;k<TK;k++)acc+=As[ty][k]*Bs[k][tx];
        __syncthreads();
    }
    if(br+ty<M&&bc+tx<N)C[(br+ty)*N+bc+tx]=T(acc);
}

static constexpr int TM=64,TN=64,TK=8,BM=16,BN=16,PAD=1,RM=TM/BM,RN=TN/BN;

template <typename T>
__global__ void gemm_vectorized_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    __shared__ float As[TM][TK+PAD],Bs[TK][TN+PAD];
    int br=blockIdx.y*TM,bc=blockIdx.x*TN,ty=threadIdx.y,tx=threadIdx.x;
    float acc[RM][RN]={};
    for(int kb=0;kb<(K+TK-1)/TK;kb++){
        for(int i=ty*BN+tx;i<(TM*TK)/4;i+=BM*BN){int m=i/(TK/4),k4=(i%(TK/4))*4,gr=br+m,gk=kb*TK+k4;if(gr<M&&gk+3<K){float4 v=reinterpret_cast<const float4*>(A+gr*K+gk)[0];As[m][k4]=v.x;As[m][k4+1]=v.y;As[m][k4+2]=v.z;As[m][k4+3]=v.w;}else for(int d=0;d<4;d++)As[m][k4+d]=(gr<M&&gk+d<K)?float(A[gr*K+gk+d]):0.f;}
        for(int i=ty*BN+tx;i<(TK*TN)/4;i+=BM*BN){int k=i/(TN/4),n4=(i%(TN/4))*4,gk=kb*TK+k,gn=bc+n4;if(gk<K&&gn+3<N){float4 v=reinterpret_cast<const float4*>(B+gk*N+gn)[0];Bs[k][n4]=v.x;Bs[k][n4+1]=v.y;Bs[k][n4+2]=v.z;Bs[k][n4+3]=v.w;}else for(int d=0;d<4;d++)Bs[k][n4+d]=(gk<K&&gn+d<N)?float(B[gk*N+gn+d]):0.f;}
        __syncthreads();
        for(int rn=0;rn<RN;rn++)for(int k=0;k<TK;k++){float bv=Bs[k][tx*RN+rn];for(int rm=0;rm<RM;rm++)acc[rm][rn]+=As[ty*RM+rm][k]*bv;}
        __syncthreads();
    }
    for(int rm=0;rm<RM;rm++)for(int rn=0;rn<RN;rn++){int gr=br+ty*RM+rm,gn=bc+tx*RN+rn;if(gr<M&&gn<N)C[gr*N+gn]=T(acc[rm][rn]);}
}

template <typename T>
__global__ void gemm_doublebuf_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    __shared__ float As0[TM][TK+PAD],As1[TM][TK+PAD],Bs0[TK][TN+PAD],Bs1[TK][TN+PAD];
    int br=blockIdx.y*TM,bc=blockIdx.x*TN,ty=threadIdx.y,tx=threadIdx.x;
    float acc[RM][RN]={};
    // Prefetch tile 0
    for(int i=ty*BN+tx;i<(TM*TK)/4;i+=BM*BN){int m=i/(TK/4),k4=(i%(TK/4))*4,gr=br+m;if(gr<M&&k4+3<K){float4 v=reinterpret_cast<const float4*>(A+gr*K+k4)[0];As0[m][k4]=v.x;As0[m][k4+1]=v.y;As0[m][k4+2]=v.z;As0[m][k4+3]=v.w;}else for(int d=0;d<4;d++)As0[m][k4+d]=(gr<M&&k4+d<K)?float(A[gr*K+k4+d]):0.f;}
    for(int i=ty*BN+tx;i<(TK*TN)/4;i+=BM*BN){int k=i/(TN/4),n4=(i%(TN/4))*4,gn=bc+n4;if(k<K&&gn+3<N){float4 v=reinterpret_cast<const float4*>(B+k*N+gn)[0];Bs0[k][n4]=v.x;Bs0[k][n4+1]=v.y;Bs0[k][n4+2]=v.z;Bs0[k][n4+3]=v.w;}else for(int d=0;d<4;d++)Bs0[k][n4+d]=(k<K&&gn+d<N)?float(B[k*N+gn+d]):0.f;}
    __syncthreads();
    for(int kb=1;kb<(K+TK-1)/TK;kb++){
        int gk_off=kb*TK;
        for(int i=ty*BN+tx;i<(TM*TK)/4;i+=BM*BN){int m=i/(TK/4),k4=(i%(TK/4))*4,gr=br+m,gk=gk_off+k4;if(gr<M&&gk+3<K){float4 v=reinterpret_cast<const float4*>(A+gr*K+gk)[0];As1[m][k4]=v.x;As1[m][k4+1]=v.y;As1[m][k4+2]=v.z;As1[m][k4+3]=v.w;}else for(int d=0;d<4;d++)As1[m][k4+d]=(gr<M&&gk+d<K)?float(A[gr*K+gk+d]):0.f;}
        for(int i=ty*BN+tx;i<(TK*TN)/4;i+=BM*BN){int k=i/(TN/4),n4=(i%(TN/4))*4,gk=gk_off+k,gn=bc+n4;if(gk<K&&gn+3<N){float4 v=reinterpret_cast<const float4*>(B+gk*N+gn)[0];Bs1[k][n4]=v.x;Bs1[k][n4+1]=v.y;Bs1[k][n4+2]=v.z;Bs1[k][n4+3]=v.w;}else for(int d=0;d<4;d++)Bs1[k][n4+d]=(gk<K&&gn+d<N)?float(B[gk*N+gn+d]):0.f;}
        // Compute from set 0, next tile loads into set 1 in parallel
        for(int rn=0;rn<RN;rn++)for(int k=0;k<TK;k++){float bv=Bs0[k][tx*RN+rn];for(int rm=0;rm<RM;rm++)acc[rm][rn]+=As0[ty*RM+rm][k]*bv;}
        __syncthreads();
        // Swap
        for(int i=ty*BN+tx;i<(TM*TK)/4;i+=BM*BN){int m=i/(TK/4),k4=(i%(TK/4))*4;As0[m][k4]=As1[m][k4];As0[m][k4+1]=As1[m][k4+1];As0[m][k4+2]=As1[m][k4+2];As0[m][k4+3]=As1[m][k4+3];}
        for(int i=ty*BN+tx;i<(TK*TN)/4;i+=BM*BN){int k=i/(TN/4),n4=(i%(TN/4))*4;Bs0[k][n4]=Bs1[k][n4];Bs0[k][n4+1]=Bs1[k][n4+1];Bs0[k][n4+2]=Bs1[k][n4+2];Bs0[k][n4+3]=Bs1[k][n4+3];}
        __syncthreads();
    }
    for(int rn=0;rn<RN;rn++)for(int k=0;k<TK;k++){float bv=Bs0[k][tx*RN+rn];for(int rm=0;rm<RM;rm++)acc[rm][rn]+=As0[ty*RM+rm][k]*bv;}
    for(int rm=0;rm<RM;rm++)for(int rn=0;rn<RN;rn++){int gr=br+ty*RM+rm,gn=bc+tx*RN+rn;if(gr<M&&gn<N)C[gr*N+gn]=T(acc[rm][rn]);}
}

// ====== Dispatch ======

Tensor gemm(const Tensor& a, const Tensor& b, bool transpose_b) {
    int M=a.size(0),K=a.size(1),N=transpose_b?b.size(0):b.size(1);
    if(a.size(1)!=(transpose_b?b.size(1):b.size(0)))throw std::runtime_error("gemm: inner dim mismatch");
    bool fp16=(a.dtype()==DType::F16);
    Tensor c({M,N},fp16?DType::F16:DType::F32,Device::CUDA);

    if(g_gemm_backend==GemmBackend::CuBLAS||fp16){
        float al=1,be=0;if(!g_handle)cublasCreate(&g_handle);
        if(fp16)cublasGemmEx(g_handle,transpose_b?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,
            N,M,K,&al,b.raw(),CUDA_R_16F,transpose_b?K:N,a.raw(),CUDA_R_16F,K,
            &be,c.raw(),CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
        else cublasSgemm_v2(g_handle,transpose_b?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,
            N,M,K,&al,b.data<float>(),transpose_b?K:N,a.data<float>(),K,&be,c.data<float>(),N);
    }else{
        if(g_gemm_backend==GemmBackend::Naive){dim3 b16(16,16),g16((N+15)/16,(M+15)/16);gemm_naive_kernel<float><<<g16,b16>>>(a.data<float>(),b.data<float>(),c.data<float>(),M,N,K);}
        else if(g_gemm_backend==GemmBackend::Tiled){dim3 b16(16,16),g16((N+15)/16,(M+15)/16);gemm_tiled_kernel<float><<<g16,b16>>>(a.data<float>(),b.data<float>(),c.data<float>(),M,N,K);}
        else if(g_gemm_backend==GemmBackend::Vectorized){dim3 bv(BN,BM),gv((N+TN-1)/TN,(M+TM-1)/TM);gemm_vectorized_kernel<float><<<gv,bv>>>(a.data<float>(),b.data<float>(),c.data<float>(),M,N,K);}
        else{dim3 bv(BN,BM),gv((N+TN-1)/TN,(M+TM-1)/TM);gemm_doublebuf_kernel<float><<<gv,bv>>>(a.data<float>(),b.data<float>(),c.data<float>(),M,N,K);}
    }
    return std::move(c);
}

// fp16 inputs → fp32 output (mixed precision via Tensor Cores)
Tensor gemm_f16f32(const Tensor& a, const Tensor& b, bool transpose_b) {
    int M=a.size(0), K=a.size(1), N=transpose_b?b.size(0):b.size(1);
    if(a.size(1)!=(transpose_b?b.size(1):b.size(0)))
        throw std::runtime_error("gemm_f16f32: inner dim mismatch");
    if(a.dtype()!=DType::F16||b.dtype()!=DType::F16)
        throw std::runtime_error("gemm_f16f32: both inputs must be fp16");

    Tensor c({M,N}, DType::F32, Device::CUDA);

    float al=1, be=0;
    if(!g_handle) cublasCreate(&g_handle);
    cublasGemmEx(g_handle,
        transpose_b?CUBLAS_OP_T:CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &al,
        b.raw(), CUDA_R_16F, transpose_b?K:N,
        a.raw(), CUDA_R_16F, K,
        &be,
        c.raw(), CUDA_R_32F, N,          // ← fp32 output!
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    return std::move(c);
}

}  // namespace ops
}  // namespace lightllm
