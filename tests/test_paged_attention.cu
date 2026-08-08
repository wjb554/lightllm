/// PagedAttention test: correctness vs CPU baseline.
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "lightllm/tensor.h"
#include "lightllm/kv_cache/block_allocator.h"
#include "lightllm/kv_cache/paged_attention.h"

using namespace lightllm;
using namespace lightllm::kv_cache;

static int passed=0,failed=0;
static void CHECK(const char* n,bool c){if(c)passed++;else{failed++;fprintf(stderr,"  FAIL: %s\n",n);}}
static bool approx(float a,float b,float t=1e-3f){return fabsf(a-b)<t;}

// CPU reference attention
static std::vector<float> attention_ref(
    const float* q,int T,int Hq,int D,
    const float* k,int Tk,int Hkv,
    const float* v,int groups)
{
    std::vector<float> out(T*Hq*D,0.f);
    float scale=1.f/sqrtf(float(D));
    for(int h=0;h<Hq;h++){int kvh=h/groups;
        for(int t=0;t<T;t++){
            std::vector<float> scores(Tk);
            float mx=-INFINITY;
            for(int s=0;s<Tk;s++){float dot=0;
                for(int d=0;d<D;d++)dot+=q[(t*Hq+h)*D+d]*k[(s*Hkv+kvh)*D+d];
                scores[s]=dot*scale;mx=fmaxf(mx,scores[s]);}
            float sm=0;for(int s=0;s<Tk;s++){scores[s]=expf(scores[s]-mx);sm+=scores[s];}
            for(int s=0;s<Tk;s++)scores[s]/=sm;
            for(int d=0;d<D;d++){float acc=0;
                for(int s=0;s<Tk;s++)acc+=scores[s]*v[(s*Hkv+kvh)*D+d];
                out[(t*Hq+h)*D+d]=acc;}
        }
    }
    return out;
}

int main() {
    int T=2,Hq=4,Hkv=2,D=64,block_size=16,num_blocks=64,max_blocks=8;
    int groups=Hq/Hkv;
    float scale=1.f/sqrtf(float(D));

    printf("=== BlockAllocator ===\n");
    BlockAllocator alloc(num_blocks,block_size,Hkv,D);
    CHECK("alloc_num_blocks",alloc.num_blocks()==64);
    CHECK("alloc_free",alloc.num_free()==64);
    CHECK("alloc_block_size",alloc.block_size()==16);

    // Allocate blocks
    int b0=alloc.allocate(0x123), b1=alloc.allocate(0x456);
    CHECK("alloc_0",b0>=0);CHECK("alloc_1",b1>=0);
    CHECK("free_62",alloc.num_free()==62);
    alloc.release(b0);alloc.release(b1);
    CHECK("free_64",alloc.num_free()==64);
    printf("  [PASS] BlockAllocator\n");

    printf("\n=== BlockTable ===\n");
    BlockTable bt(max_blocks);
    bt.append(3);bt.append(7);bt.append(12);
    CHECK("bt_size",bt.size()==3);
    CHECK("bt_0",bt[0]==3);CHECK("bt_1",bt[1]==7);CHECK("bt_2",bt[2]==12);
    printf("  [PASS] BlockTable\n");

    printf("\n=== PagedAttention Correctness ===\n");
    // Create K/V in physical blocks
    int seq_len=24; // 1.5 blocks of tokens
    std::vector<float> kv_data(num_blocks*block_size*Hkv*D,0.f);
    for(int i=0;i<num_blocks*block_size*Hkv*D;i++) kv_data[i]=float(i%100-50)/100.f;

    // Manually copy K/V data into allocator storage
    cudaMemcpy((float*)alloc.k_storage_raw(),kv_data.data(),
        num_blocks*block_size*Hkv*D*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy((float*)alloc.v_storage_raw(),kv_data.data(),
        num_blocks*block_size*Hkv*D*sizeof(float),cudaMemcpyHostToDevice);

    // Set up block table: sequence 0 uses blocks 0 and 1, sequence 1 uses block 2
    // Need this on GPU for the kernel
    std::vector<int> h_bt(T*max_blocks,-1);
    h_bt[0*max_blocks+0]=0;h_bt[0*max_blocks+1]=1; // token 0: blocks 0,1
    h_bt[1*max_blocks+0]=2;                          // token 1: block 2
    int* d_bt;cudaMalloc(&d_bt,T*max_blocks*sizeof(int));
    cudaMemcpy(d_bt,h_bt.data(),T*max_blocks*sizeof(int),cudaMemcpyHostToDevice);

    std::vector<int> h_sl={seq_len,8}; // seq 0 has 24 tokens, seq 1 has 8
    int* d_sl;cudaMalloc(&d_sl,T*sizeof(int));
    cudaMemcpy(d_sl,h_sl.data(),T*sizeof(int),cudaMemcpyHostToDevice);

    // Q: random values
    Tensor q({T,Hq,D},DType::F32,Device::CUDA);
    std::vector<float> h_q(T*Hq*D);
    for(size_t i=0;i<h_q.size();i++)h_q[i]=float(i%100-50)/100.f;
    q.copy_from(h_q.data(),h_q.size()*sizeof(float));

    // Run PagedAttention
    auto out=paged_attention(q,d_bt,max_blocks,d_sl,alloc);
    cudaDeviceSynchronize();
    auto h_out=std::vector<float>(out.numel());
    out.copy_to(h_out.data(),out.numel()*sizeof(float));

    // CPU reference
    // Reconstruct contiguous K/V from paged storage
    std::vector<float> k_cont0(seq_len*Hkv*D), v_cont0(seq_len*Hkv*D);
    for(int pos=0;pos<seq_len;pos++){
        int b=pos/block_size,offset=pos%block_size;
        int phys=h_bt[0*max_blocks+b];
        for(int h=0;h<Hkv;h++)for(int d=0;d<D;d++){
            k_cont0[pos*Hkv*D+h*D+d]=kv_data[phys*block_size*Hkv*D+offset*Hkv*D+h*D+d];
            v_cont0[pos*Hkv*D+h*D+d]=kv_data[phys*block_size*Hkv*D+offset*Hkv*D+h*D+d];
        }
    }
    int Tk1=8;
    std::vector<float> k_cont1(Tk1*Hkv*D),v_cont1(Tk1*Hkv*D);
    for(int pos=0;pos<Tk1;pos++){
        int b=pos/block_size,offset=pos%block_size;
        int phys=h_bt[1*max_blocks+b];
        for(int h=0;h<Hkv;h++)for(int d=0;d<D;d++){
            k_cont1[pos*Hkv*D+h*D+d]=kv_data[phys*block_size*Hkv*D+offset*Hkv*D+h*D+d];
            v_cont1[pos*Hkv*D+h*D+d]=kv_data[phys*block_size*Hkv*D+offset*Hkv*D+h*D+d];
        }
    }

    auto ref0=attention_ref(h_q.data(),1,Hq,D,k_cont0.data(),seq_len,Hkv,v_cont0.data(),groups);
    for(int i=0;i<Hq*D;i++) CHECK("paged0",approx(h_out[i],ref0[i],2e-2f));
    auto ref1=attention_ref(h_q.data()+Hq*D,1,Hq,D,k_cont1.data(),Tk1,Hkv,v_cont1.data(),groups);
    for(int i=0;i<Hq*D;i++) CHECK("paged1",approx(h_out[Hq*D+i],ref1[i],2e-2f));

    cudaFree(d_bt);cudaFree(d_sl);
    printf("  [PASS] PagedAttention vs CPU\n");

    printf("\n%s: %d passed, %d failed\n",failed?"SOME FAILURES":"ALL PASSED",passed,failed);
    return failed?1:0;
}
