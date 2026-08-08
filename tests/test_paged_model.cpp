/// PagedAttention infrastructure test — BlockAllocator + BlockTable + model integration.
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "lightllm/tensor.h"
#include "lightllm/kv_cache/block_allocator.h"
#include "lightllm/model/model_loader.h"
#include "lightllm/model/model_config.h"
#include "lightllm/ops/norm.h"
#include "lightllm/ops/rope.h"
#include "lightllm/ops/gemm.h"
#include "lightllm/ops/elementwise.h"
using namespace lightllm;using namespace lightllm::model;using namespace lightllm::kv_cache;using namespace lightllm::ops;
static int passed=0,failed=0;
static void CHECK(const char* n,bool c){if(c)passed++;else{failed++;fprintf(stderr,"  FAIL: %s\n",n);}}
static std::vector<float> bf16f32(const std::vector<char>& r){std::vector<float> f(r.size()/2);for(size_t i=0;i<f.size();i++){uint16_t b=((uint16_t*)r.data())[i];uint32_t u=b<<16;f[i]=*(float*)&u;}return f;}
static Tensor load_f32(SafetensorsLoader& ldr,const std::string& n){auto* info=ldr.get_info(n);Tensor cpu=ldr.load_tensor(n);std::vector<char> raw(cpu.nbytes());cpu.copy_to(raw.data(),cpu.nbytes());auto f32=bf16f32(raw);Tensor gpu(info->shape,DType::F32,Device::CUDA);gpu.copy_from(f32.data(),f32.size()*sizeof(float));return gpu;}

int main(){
    const char* mdl="models/qwen2.5-0.5b";char buf[512];
    snprintf(buf,sizeof(buf),"%s/config.json",mdl);auto cfg=parse_config(buf);
    snprintf(buf,sizeof(buf),"%s/model.safetensors",mdl);SafetensorsLoader loader(buf);
    int D=cfg.hidden_size,Hq=cfg.num_attention_heads,Hkv=cfg.num_key_value_heads,hd=cfg.head_dim;
    printf("=== Paged KV Cache Test ===\nD=%d Hq=%d Hkv=%d hd=%d\n",D,Hq,Hkv,hd);

    // Init allocator
    int bs=16,nb=128;
    BlockAllocator alloc(nb,bs,Hkv,hd);
    CHECK("init",alloc.num_free()==128&&alloc.block_size()==16);

    // Allocate + release cycle
    int a0=alloc.allocate(0xABCD);
    CHECK("alloc0",a0>=0);
    int a1=alloc.allocate(0x1234);
    CHECK("alloc1",a1>=0&&a1!=a0);
    alloc.release(a0);alloc.release(a1);
    CHECK("free_all",alloc.num_free()==128);

    // Prefix cache hit (while still allocated)
    int p0=alloc.allocate(0xBEEF);
    int p1=alloc.find_by_hash(0xBEEF);
    CHECK("prefix_found",p1==p0); // found while still allocated
    alloc.release(p0);
    CHECK("prefix_cleared",alloc.find_by_hash(0xBEEF)==-1); // cleared after release

    // Block table test
    BlockTable bt(32);
    bt.append(5);bt.append(10);bt.append(15);
    CHECK("bt_size",bt.size()==3);
    CHECK("bt_idx",bt[1]==10);

    // Load a small batch of weights
    auto a_n=load_f32(loader,"model.layers.0.input_layernorm.weight");
    CHECK("weights_ok",a_n.size(0)==D);

    printf("\n%s: %d passed, %d failed\n",failed?"SOME FAILURES":"ALL PASSED",passed,failed);
    return failed?1:0;
}
