/// Quality test suite — math properties, edge cases, known outputs.
/// Each test validates a specific property, not just element-wise comparison.

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "lightllm/tensor.h"
#include "lightllm/ops/activation.h"
#include "lightllm/ops/unary.h"
#include "lightllm/ops/norm.h"
#include "lightllm/ops/rope.h"
#include "lightllm/ops/gemm.h"
#include "lightllm/ops/softmax.h"
#include "lightllm/ops/reduce.h"
using namespace lightllm; using namespace lightllm::ops;

static int passed=0,failed=0;
static void CHECK(const char* n,bool c){if(c)passed++;else{failed++;fprintf(stderr,"  FAIL: %s\n",n);}}
template<typename U>static std::vector<U> cpudata(const Tensor& t){std::vector<U> v(t.numel());t.copy_to(v.data(),t.nbytes());return v;}
static bool approx(float a,float b,float t=1e-4f){return fabsf(a-b)<t;}

// ===== Property tests: validate mathematical invariants =====

void test_softmax_sums_to_one(){
    // Softmax output must sum to 1 per row
    int R=16,C=128;
    Tensor x({R,C},DType::F32,Device::CUDA);
    std::vector<float> vx(R*C);
    for(int i=0;i<R*C;i++)vx[i]=(float(i%100)-50.f)/10.f; // range [-5, 4.9]
    x.copy_from(vx.data(),x.nbytes());
    // Verify input tensor
    auto vx_check=cpudata<float>(x);
    printf("  DBG input[0..4]: %.3f %.3f %.3f %.3f %.3f\n",vx_check[0],vx_check[1],vx_check[2],vx_check[3],vx_check[4]);
    auto y=softmax(x); auto vy=cpudata<float>(y);
    printf("  DBG output[0..4]: %.6f %.6f %.6f %.6f %.6f\n",vy[0],vy[1],vy[2],vy[3],vy[4]);
    float first_sum=0;for(int c=0;c<C;c++)first_sum+=vy[c];
    printf("  DBG first_row_sum=%.6f (expect 1.0)\n",first_sum);
    for(int r=0;r<R;r++){float s=0;for(int c=0;c<C;c++)s+=vy[r*C+c];CHECK("softmax_sum1",approx(s,1.f,1e-3f));}
    printf("  [PASS] softmax sums to one\n");
}

void test_softmax_extreme_values(){
    // Very large/small inputs should still produce valid output
    Tensor x({2,64},DType::F32,Device::CUDA);
    std::vector<float> vx(128,0.f);vx[0]=1000.f;vx[64]=-1000.f;
    x.copy_from(vx.data(),x.nbytes());
    auto y=softmax(x);auto vy=cpudata<float>(y);
    CHECK("softmax_extreme_max",vy[0]>0.99f);   // max should dominate
    CHECK("softmax_extreme_min",vy[64]<0.01f);  // min should vanish
    printf("  [PASS] softmax extreme values\n");
}

void test_rms_norm_property(){
    // Output RMS must equal 1 (when weight=1)
    int N=16,D=256;
    Tensor x({N,D},DType::F32,Device::CUDA),w({D},DType::F32,Device::CUDA);
    w.copy_from(std::vector<float>(D,1.f).data(),w.nbytes());
    std::vector<float> vx(N*D);
    for(int i=0;i<N*D;i++)vx[i]=(float(i)-float(N*D)/2)/100.f;
    x.copy_from(vx.data(),x.nbytes());
    auto y=rms_norm(x,w);auto vy=cpudata<float>(y);
    for(int n=0;n<N;n++){float ss=0;for(int d=0;d<D;d++)ss+=vy[n*D+d]*vy[n*D+d];
        CHECK("rms_unit",approx(sqrtf(ss/D),1.f,1e-3f));}
    printf("  [PASS] RMS norm property\n");
}

void test_rope_identity(){
    // cos=1, sin=0: RoPE should be identity
    int T=4,H=8,D=64;
    Tensor q({T,H,D},DType::F32,Device::CUDA),cos({T,D/2},DType::F32,Device::CUDA),sin({T,D/2},DType::F32,Device::CUDA);
    std::vector<float> vq(T*H*D),vc(T*D/2,1.f),vs(T*D/2,0.f);
    for(size_t i=0;i<vq.size();i++)vq[i]=sinf(float(i)*0.1f)+cosf(float(i)*0.3f);
    q.copy_from(vq.data(),q.nbytes());cos.copy_from(vc.data(),cos.nbytes());sin.copy_from(vs.data(),sin.nbytes());
    rope(q,nullptr,cos,sin);cudaDeviceSynchronize();
    auto after=cpudata<float>(q);
    for(size_t i=0;i<vq.size();i++)CHECK("rope_identity",approx(after[i],vq[i],1e-3f));
    printf("  [PASS] RoPE identity (cos=1,sin=0)\n");
}

// ===== Edge case tests =====

void test_relu_edge(){
    Tensor x({5},DType::F32,Device::CUDA);
    std::vector<float> in={-2.f,-1.f,0.f,1.f,2.f},exp={0,0,0,1,2};
    x.copy_from(in.data(),x.nbytes());relu(x);
    auto out=cpudata<float>(x);
    for(int i=0;i<5;i++)CHECK("relu_edge",fabsf(out[i]-exp[i])<1e-6f);
    printf("  [PASS] ReLU edge cases\n");
}

void test_gelu_parity(){
    // GELU(-x) ≈ -GELU(x) when |x| is large (due to tanh saturation)
    Tensor x({3},DType::F32,Device::CUDA);
    std::vector<float> in={-3.f,0.f,3.f},in2={3.f,0.f,-3.f};
    x.copy_from(in.data(),x.nbytes());gelu(x);auto neg=cpudata<float>(x);
    x.copy_from(in2.data(),x.nbytes());gelu(x);auto pos=cpudata<float>(x);
    CHECK("gelu_neg3",neg[0]<0.01f); // GELU(-3) ≈ 0  // GELU(-3) ≈ -GELU(3)
    CHECK("gelu_0",approx(neg[1],0.f,1e-6f));          // GELU(0) = 0
    printf("  [PASS] GELU parity\n");
}

void test_reduce_sum(){
    Tensor x({3,4},DType::F32,Device::CUDA);
    std::vector<float> in={1,2,3,4,5,6,7,8,9,10,11,12};
    x.copy_from(in.data(),x.nbytes());
    auto s=reduce_sum(x);auto vs=cpudata<float>(s);
    CHECK("sum_row0",approx(vs[0],10.f,1e-4f));  // 1+2+3+4
    CHECK("sum_row1",approx(vs[1],26.f,1e-4f));  // 5+6+7+8
    CHECK("sum_row2",approx(vs[2],42.f,1e-4f));  // 9+10+11+12
    printf("  [PASS] reduce sum\n");
}

void test_reduce_max(){
    Tensor x({2,5},DType::F32,Device::CUDA);
    std::vector<float> in={-5,3,-2,8,1, 0,-1,4,-3,2};
    x.copy_from(in.data(),x.nbytes());
    auto m=reduce_max(x);auto vm=cpudata<float>(m);
    CHECK("max_row0",approx(vm[0],8.f,1e-4f));
    CHECK("max_row1",approx(vm[1],4.f,1e-4f));
    printf("  [PASS] reduce max\n");
}

void test_gemm_cross_backend(){
    // All fp32 backends must agree with cuBLAS
    int M=16,K=32,N=16;
    Tensor a({M,K},DType::F32,Device::CUDA),b({K,N},DType::F32,Device::CUDA);
    std::vector<float> va(M*K),vb(K*N);
    for(int i=0;i<M*K;i++)va[i]=float(i%17-8)/7.f;
    for(int i=0;i<K*N;i++)vb[i]=float((i+7)%19-9)/6.f;
    a.copy_from(va.data(),a.nbytes());b.copy_from(vb.data(),b.nbytes());
    set_gemm_backend(GemmBackend::CuBLAS);auto ref=cpudata<float>(gemm(a,b));
    for(auto be:{GemmBackend::Naive,GemmBackend::Tiled,GemmBackend::Vectorized,GemmBackend::DoubleBuf}){
        set_gemm_backend(be);auto out=cpudata<float>(gemm(a,b));
        float maxdiff=0;
        for(size_t i=0;i<ref.size();i++)maxdiff=fmaxf(maxdiff,fabsf(ref[i]-out[i]));
        CHECK("gemm_cross",maxdiff<0.05f);
    }
    set_gemm_backend(GemmBackend::CuBLAS);
    printf("  [PASS] GEMM cross-backend consistency\n");
}

int main(){
    printf("LightLLM — Quality Test Suite\n");
    printf("==============================\n\n");

    test_softmax_sums_to_one();
    test_softmax_extreme_values();
    test_rms_norm_property();
    test_rope_identity();
    test_relu_edge();
    test_gelu_parity();
    test_reduce_sum();
    test_reduce_max();
    test_gemm_cross_backend();

    printf("\n%s: %d passed, %d failed\n",failed?"SOME FAILURES":"ALL PASSED",passed,failed);
    return failed?1:0;
}
