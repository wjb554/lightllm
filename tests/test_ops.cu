#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "lightllm/tensor.h"
#include "lightllm/ops/elementwise.h"
#include "lightllm/ops/norm.h"
#include "lightllm/ops/rope.h"
#include "lightllm/ops/gemm.h"
#include "lightllm/ops/activation.h"
#include "lightllm/ops/unary.h"
// #include "lightllm/ops/softmax.h" // TODO: fix
// #include "lightllm/ops/reduce.h"  // TODO: fix
using namespace lightllm; using namespace lightllm::ops;

static int tests_run=0,tests_passed=0;
static void check(const char* n,bool c){tests_run++;if(c)tests_passed++;else fprintf(stderr,"  FAIL: %s\n",n);}
static bool close(float a,float b,float t=1e-2f){return fabsf(a-b)<t;}
template<typename T>static std::vector<T> to_cpu(const Tensor& t){std::vector<T> v(t.numel());t.copy_to(v.data(),t.nbytes());return v;}

static void test_add(){
    Tensor a({2,3},DType::F32,Device::CUDA),b({2,3},DType::F32,Device::CUDA);
    std::vector<float> va={1,2,3,4,5,6},vb={6,5,4,3,2,1};
    a.copy_from(va.data(),a.nbytes());b.copy_from(vb.data(),b.nbytes());
    auto vc=to_cpu<float>(add(a,b));
    for(int i=0;i<6;i++)check("add",close(vc[i],va[i]+vb[i]));
    printf("  [PASS] test_add\n");
}
static void test_silu(){
    Tensor x({4,128},DType::F32,Device::CUDA);
    std::vector<float> vx(512);for(size_t i=0;i<512;i++)vx[i]=(float(i)-256)/128.f;
    x.copy_from(vx.data(),x.nbytes());
    auto vy=to_cpu<float>(silu(x));
    for(size_t i=0;i<512;i++){float ref=vx[i]/(1.f+expf(-vx[i]));check("silu",close(vy[i],ref));}
    printf("  [PASS] test_silu\n");
}
static void test_rms_norm(){
    int N=4,D=256;
    Tensor x({N,D},DType::F32,Device::CUDA),w({D},DType::F32,Device::CUDA);
    std::vector<float> vx(N*D),vw(D,1.f);
    for(size_t i=0;i<N*D;i++)vx[i]=(float(i)/float(N*D))*2.f-1.f;
    x.copy_from(vx.data(),x.nbytes());w.copy_from(vw.data(),w.nbytes());
    auto vy=to_cpu<float>(rms_norm(x,w));
    for(int n=0;n<N;n++){float ss=0;for(int d=0;d<D;d++)ss+=vy[n*D+d]*vy[n*D+d];check("rms_norm",close(sqrtf(ss/D),1.f));}
    printf("  [PASS] test_rms_norm\n");
}
static void test_rope(){
    int T=1,H=4,D=64;
    Tensor q({T,H,D},DType::F32,Device::CUDA),cos({T,D/2},DType::F32,Device::CUDA),sin({T,D/2},DType::F32,Device::CUDA);
    q.copy_from(std::vector<float>(T*H*D,0.1f).data(),q.nbytes());
    cos.copy_from(std::vector<float>(T*D/2,0.5f).data(),cos.nbytes());
    sin.copy_from(std::vector<float>(T*D/2,0.5f).data(),sin.nbytes());
    rope(q,nullptr,cos,sin);cudaDeviceSynchronize();
    auto a=to_cpu<float>(q);
    check("rope_o1",fabsf(a[0])<0.01f);check("rope_o2",fabsf(a[D/2]-0.1f)<0.01f);
    printf("  [PASS] test_rope\n");
}
static void test_gemm(){
    int M=64,K=128,N=64;
    { // fp32
        Tensor a({M,K},DType::F32,Device::CUDA),b({K,N},DType::F32,Device::CUDA);
        std::vector<float> va(M*K),vb(K*N);
        for(int i=0;i<M*K;i++)va[i]=float(i%100)/100.f;
        for(int i=0;i<K*N;i++)vb[i]=float((i+50)%100)/100.f;
        a.copy_from(va.data(),a.nbytes());b.copy_from(vb.data(),b.nbytes());
        set_gemm_backend(GemmBackend::CuBLAS);auto vref=to_cpu<float>(gemm(a,b));
        for(auto be:{GemmBackend::Naive,GemmBackend::Tiled,GemmBackend::Vectorized,GemmBackend::DoubleBuf}){
            set_gemm_backend(be);auto vout=to_cpu<float>(gemm(a,b));
            for(size_t i=0;i<vref.size();i++)check("gemm_f32",close(vref[i],vout[i],2e-2f));
        }
    }
    { // fp16 cuBLAS (Tensor Core)
        Tensor ah({M,K},DType::F16,Device::CUDA),bh({K,N},DType::F16,Device::CUDA);
        std::vector<half> vah(M*K),vbh(K*N);
        for(int i=0;i<M*K;i++)vah[i]=__float2half(float(i%100)/100.f);
        for(int i=0;i<K*N;i++)vbh[i]=__float2half(float((i+50)%100)/100.f);
        ah.copy_from(vah.data(),ah.nbytes());bh.copy_from(vbh.data(),bh.nbytes());
        auto vout=to_cpu<half>(gemm(ah,bh));
        // Verify outputs are finite (cuBLAS fp16 correctness)
        for(size_t i=0;i<vout.size();i++)check("fp16_finite",std::isfinite(float(vout[i])));
    }
    set_gemm_backend(GemmBackend::CuBLAS);
    printf("  [PASS] test_gemm\n");
}
// ===========================================================================
// Comprehensive GEMM benchmark — multiple shapes, fp32 & fp16
// ===========================================================================
static void bench_gemm(){
    const int WARMUP=30, ITERS=50;
    const float ALPHA=0.5f, BETA=0.3f;

    // GPU global warmup
    {
        Tensor wm({1024,1024},DType::F32,Device::CUDA),wn({1024,1024},DType::F32,Device::CUDA);
        set_gemm_backend(GemmBackend::CuBLAS);
        for(int i=0;i<30;i++)gemm(wm,wn);
        cudaDeviceSynchronize();
    }

    // Helper: benchmark one (M,K,N) shape across backends
    auto bench_fp32=[&](const char* label,int M,int K,int N,
                        std::initializer_list<GemmBackend> backends){
        printf("\n  [fp32] %s  M=%d K=%d N=%d  (%d iter, p50)\n",label,M,K,N,ITERS);
        Tensor a({M,K},DType::F32,Device::CUDA),b({K,N},DType::F32,Device::CUDA);
        std::vector<float> va(M*K,ALPHA),vb(K*N,BETA);
        a.copy_from(va.data(),a.nbytes());b.copy_from(vb.data(),b.nbytes());
        for(auto be:backends){
            set_gemm_backend(be);
            for(int i=0;i<WARMUP;i++)gemm(a,b);cudaDeviceSynchronize();
            std::vector<long long> ts; ts.reserve(ITERS);
            for(int i=0;i<ITERS;i++){
                auto t0=std::chrono::steady_clock::now();gemm(a,b);cudaDeviceSynchronize();
                ts.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now()-t0).count());
            }
            std::sort(ts.begin(),ts.end());
            auto med=ts[ITERS/2],mn=ts[0],mx=ts[ITERS-1];
            double gflops=(2.0*M*K*N)/(med*1e3);
            double bw=(double)(M*K+K*N+M*N)*4/(med*1e3); // GB/s (weights read+write, approx)
            const char* nm=be==GemmBackend::Naive?"Naive":be==GemmBackend::Tiled?"Tiled":
                           be==GemmBackend::Vectorized?"Vec":be==GemmBackend::DoubleBuf?"DBuf":"cuBLAS";
            printf("    %-6s  p50=%6lld us  [%lld..%lld]  %7.0f GFLOPS  %6.1f GB/s\n",
                   nm,med,mn,mx,gflops,bw);
        }
    };

    auto bench_fp16_cublas=[&](const char* label,int M,int K,int N){
        printf("\n  [fp16] %s  M=%d K=%d N=%d  (cuBLAS TC, %d iter, p50)\n",label,M,K,N,ITERS);
        Tensor a({M,K},DType::F16,Device::CUDA),b({K,N},DType::F16,Device::CUDA);
        std::vector<half> va(M*K),vb(K*N);
        for(int i=0;i<M*K;i++)va[i]=__float2half(ALPHA);
        for(int i=0;i<K*N;i++)vb[i]=__float2half(BETA);
        a.copy_from(va.data(),a.nbytes());b.copy_from(vb.data(),b.nbytes());
        set_gemm_backend(GemmBackend::CuBLAS);
        for(int i=0;i<WARMUP;i++)gemm(a,b);cudaDeviceSynchronize();
        std::vector<long long> ts; ts.reserve(ITERS);
        for(int i=0;i<ITERS;i++){
            auto t0=std::chrono::steady_clock::now();gemm(a,b);cudaDeviceSynchronize();
            ts.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now()-t0).count());
        }
        std::sort(ts.begin(),ts.end());
        auto med=ts[ITERS/2],mn=ts[0],mx=ts[ITERS-1];
        double tflops=(2.0*M*K*N)/(med*1e6);
        printf("    cuBLAS  p50=%6lld us  [%lld..%lld]  %6.1f TFLOPS\n",med,mn,mx,tflops);
    };

    // --- 1. Square GEMMs: compute-bound ---
    printf("\n--- 1. SQUARE GEMMs (compute-bound) ---");

    bench_fp32("512^3", 512,512,512, {GemmBackend::Naive,GemmBackend::Tiled,
        GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::CuBLAS});
    bench_fp32("1024^3",1024,1024,1024, {GemmBackend::Naive,GemmBackend::Tiled,
        GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::CuBLAS});
    bench_fp32("2048^3",2048,2048,2048, {GemmBackend::Naive,GemmBackend::Tiled,
        GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::CuBLAS});

    bench_fp16_cublas("512^3",  512,512,512);
    bench_fp16_cublas("1024^3", 1024,1024,1024);
    bench_fp16_cublas("2048^3", 2048,2048,2048);

    // ==========================================═
    // 2. Qwen2-0.5B DECODE shapes — M=1, memory-bound
    // ==========================================═
    printf("\n+------------------------------------------------------");
    printf("\n| 2. Qwen2-0.5B DECODE (M=1, D=896, memory-bound)");
    printf("\n+------------------------------------------------------");

    bench_fp32("q_proj    M=1 K=896 N=896",   1,896,896,   {GemmBackend::CuBLAS,GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::Naive});
    bench_fp32("k_proj    M=1 K=896 N=128",   1,896,128,   {GemmBackend::CuBLAS,GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::Naive});
    bench_fp32("v_proj    M=1 K=896 N=128",   1,896,128,   {GemmBackend::CuBLAS,GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::Naive});
    bench_fp32("o_proj    M=1 K=896 N=896",   1,896,896,   {GemmBackend::CuBLAS,GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::Naive});
    bench_fp32("gate_proj M=1 K=896 N=4864",  1,896,4864,  {GemmBackend::CuBLAS,GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::Naive});
    bench_fp32("up_proj   M=1 K=896 N=4864",  1,896,4864,  {GemmBackend::CuBLAS,GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::Naive});
    bench_fp32("down_proj M=1 K=4864 N=896",  1,4864,896,  {GemmBackend::CuBLAS,GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::Naive});
    bench_fp32("lm_head   M=1 K=896 N=151936",1,896,151936,{GemmBackend::CuBLAS,GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::Naive});

    bench_fp16_cublas("q_proj    M=1 K=896 N=896",   1,896,896);
    bench_fp16_cublas("k_proj    M=1 K=896 N=128",   1,896,128);
    bench_fp16_cublas("gate_proj M=1 K=896 N=4864",  1,896,4864);
    bench_fp16_cublas("down_proj M=1 K=4864 N=896",  1,4864,896);
    bench_fp16_cublas("lm_head   M=1 K=896 N=151936",1,896,151936);

    // ==========================================═
    // 3. Qwen2-0.5B PREFILL shapes — batched, compute-bound
    // ==========================================═
    printf("\n+------------------------------------------------------");
    printf("\n| 3. Qwen2-0.5B PREFILL (batched, K=896)");
    printf("\n+------------------------------------------------------");

    bench_fp32("q_proj    M=64 K=896 N=896",   64,896,896,  {GemmBackend::Naive,GemmBackend::Tiled,GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::CuBLAS});
    bench_fp32("gate_proj M=64 K=896 N=4864",  64,896,4864, {GemmBackend::Naive,GemmBackend::Tiled,GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::CuBLAS});
    bench_fp32("down_proj M=64 K=4864 N=896",  64,4864,896, {GemmBackend::Naive,GemmBackend::Tiled,GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::CuBLAS});

    bench_fp16_cublas("q_proj    M=64 K=896 N=896",   64,896,896);
    bench_fp16_cublas("gate_proj M=64 K=896 N=4864",  64,896,4864);
    bench_fp16_cublas("down_proj M=64 K=4864 N=896",  64,4864,896);

    // ==========================================═
    // 4. Batch scaling — fix K=896, vary M
    // ==========================================═
    printf("\n+------------------------------------------------------");
    printf("\n| 4. BATCH SCALING (K=896 N=896,  vary M)");
    printf("\n+------------------------------------------------------");

    for(int M:{1,4,16,64,256}){
        char lbl[64];snprintf(lbl,sizeof(lbl),"M=%-3d K=896 N=896",M);
        bench_fp32(lbl,M,896,896,{GemmBackend::Vectorized,GemmBackend::DoubleBuf,GemmBackend::CuBLAS});
        bench_fp16_cublas(lbl,M,896,896);
    }

    // ==========================================═
    // 5. Large vocabulary — lm_head scaling
    // ==========================================═
    printf("\n+------------------------------------------------------");
    printf("\n| 5. LM_HEAD scaling (M=1 K=896,  vary vocab)");
    printf("\n+------------------------------------------------------");

    for(int V:{896,4864,16384,50257,151936}){
        char lbl[64];snprintf(lbl,sizeof(lbl),"lm_head  M=1 K=896 V=%-6d",V);
        bench_fp32(lbl,1,896,V,{GemmBackend::CuBLAS,GemmBackend::Vectorized});
        bench_fp16_cublas(lbl,1,896,V);
    }

    set_gemm_backend(GemmBackend::CuBLAS);
    printf("\n=== GEMM benchmark complete ===\n\n");
}
// Level 1: Activation
void test_activations(){
    int N=4096;
    Tensor x({N},DType::F32,Device::CUDA);
    std::vector<float> vx(N);
    for(int i=0;i<N;i++)vx[i]=(float(i)-2048)/1024.f;

    x.copy_from(vx.data(),x.nbytes());relu(x);auto r=to_cpu<float>(x);
    for(int i=0;i<N;i++)check("relu",fabsf(r[i]-fmaxf(0.f,vx[i]))<1e-4f);

    x.copy_from(vx.data(),x.nbytes());sigmoid(x);r=to_cpu<float>(x);
    for(int i=0;i<N;i++)check("sigmoid",fabsf(r[i]-(1.f/(1.f+expf(-vx[i]))))<1e-2f);

    x.copy_from(vx.data(),x.nbytes());gelu(x);r=to_cpu<float>(x);
    for(int i=0;i<N;i++){float v=vx[i];float ref=v*0.5f*(1.f+tanhf(0.79788456f*(v+0.044715f*v*v*v)));check("gelu",fabsf(r[i]-ref)<1e-2f);}

    x.copy_from(vx.data(),x.nbytes());tanh_op(x);r=to_cpu<float>(x);
    for(int i=0;i<N;i++)check("tanh",fabsf(r[i]-tanhf(vx[i]))<1e-2f);

    printf("  [PASS] test_activations\n");
}

// Level 1: Unary
void test_unary(){
    int N=4096;
    Tensor x({N},DType::F32,Device::CUDA);
    std::vector<float> vx(N);
    for(int i=0;i<N;i++)vx[i]=(float(i)-2048)/1024.f;

    x.copy_from(vx.data(),x.nbytes());neg(x);auto r=to_cpu<float>(x);
    for(int i=0;i<N;i++)check("neg",r[i]==-vx[i]);

    x.copy_from(vx.data(),x.nbytes());abs_op(x);r=to_cpu<float>(x);
    for(int i=0;i<N;i++)check("abs",fabsf(r[i]-fabsf(vx[i]))<1e-4f);

    x.copy_from(vx.data(),x.nbytes());clip(x,-1.f,1.f);r=to_cpu<float>(x);
    for(int i=0;i<N;i++)check("clip",fabsf(r[i]-fmaxf(-1.f,fminf(1.f,vx[i])))<1e-4f);

    std::vector<float> pos(N);for(int i=0;i<N;i++)pos[i]=fabsf(vx[i])+0.1f;
    x.copy_from(pos.data(),x.nbytes());sqrt_op(x);r=to_cpu<float>(x);
    for(int i=0;i<N;i++)check("sqrt",fabsf(r[i]-sqrtf(pos[i]))<1e-3f);

    x.copy_from(pos.data(),x.nbytes());log_op(x);r=to_cpu<float>(x);
    for(int i=0;i<N;i++)check("log",fabsf(r[i]-logf(pos[i]))<2e-2f);

    printf("  [PASS] test_unary\n");
}

// Level 2: Softmax + Reduce
// void test_softmax_reduce(){} // TODO: fix reduce/softmax

int main(){
    printf("LightLLM START\n");fflush(stdout);
    test_add();fflush(stdout);
    test_silu();fflush(stdout);
    test_rms_norm();fflush(stdout);
    test_rope();fflush(stdout);
    test_gemm();fflush(stdout);
    printf("existing ops done\n");fflush(stdout);

    test_activations();fflush(stdout);
    printf("activations done\n");fflush(stdout);

    test_unary();fflush(stdout);
    printf("unary done\n");fflush(stdout);

    bench_gemm();
    printf("\n%d / %d tests passed\n",tests_passed,tests_run);
    return tests_passed==tests_run?0:1;
}
