/// End-to-end text generation with Qwen2.5-0.5B.
/// Prefill + Decode loop with contiguous KV cache, CPU attention baseline.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

#include "lightllm/tensor.h"
#include "lightllm/model/model_loader.h"
#include "lightllm/model/model_config.h"
#include "lightllm/ops/norm.h"
#include "lightllm/ops/rope.h"
#include "lightllm/ops/gemm.h"
#include "lightllm/ops/elementwise.h"
#include "lightllm/ops/sampling.h"
#include "lightllm/ops/lm_head.h"
#include <random>

using namespace lightllm;
using namespace lightllm::model;
using namespace lightllm::ops;

// ---- helpers ----
static std::vector<float> bf16_to_f32(const std::vector<char>& raw){
    size_t n=raw.size()/2;std::vector<float> f(n);
    for(size_t i=0;i<n;i++){uint16_t b=((const uint16_t*)raw.data())[i];uint32_t u=b<<16;f[i]=*(float*)&u;}
    return f;
}
static Tensor load_f32(SafetensorsLoader& loader, const std::string& name){
    Tensor cpu=loader.load_tensor(name);
    std::vector<char> raw(cpu.nbytes());cpu.copy_to(raw.data(),cpu.nbytes());
    auto f32=bf16_to_f32(raw);
    auto* info=loader.get_info(name);
    Tensor gpu(info->shape,DType::F32,Device::CUDA);
    gpu.copy_from(f32.data(),f32.size()*sizeof(float));
    return gpu;
}
// ---- GPU Attention kernels (3-stage: QK^T → softmax → S@V) ----
// All data stays on GPU. Intermediate scores stored in global memory.

// Stage 1: Q·K^T — one block per (token, Q-head)
__global__ void attn_qk_kernel(
    float* scores, const float* q, const float* k,
    int TQ, int Hq, int TK, int Hkv, int D, int groups, float scale)
{
    int t=blockIdx.x, h=blockIdx.y, kvh=h/groups, tid=threadIdx.x;
    for(int s=tid;s<TK;s+=blockDim.x){
        float dot=0;
        for(int d=0;d<D;d++)dot+=q[(t*Hq+h)*D+d]*k[(s*Hkv+kvh)*D+d];
        scores[(t*Hq+h)*TK+s]=dot*scale;
    }
}

// Stage 2: Softmax along last dim (reuses the same softmax code pattern)
__global__ void attn_softmax_kernel(float* scores,int rows,int cols){
    int row=blockIdx.x,tid=threadIdx.x;
    __shared__ float smem[64];
    int wid=tid/32,lid=tid%32,nw=blockDim.x/32;
    float mx=-1e38f;
    for(int i=tid;i<cols;i+=blockDim.x)mx=fmaxf(mx,scores[row*cols+i]);
    for(int o=16;o>0;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));
    if(lid==0)smem[wid]=mx;__syncthreads();
    if(tid<32){float v=(tid<nw)?smem[tid]:-1e38f;for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,o));if(tid==0)smem[0]=v;}__syncthreads();
    mx=smem[0];float sm=0;
    for(int i=tid;i<cols;i+=blockDim.x){float v=expf(scores[row*cols+i]-mx);scores[row*cols+i]=v;sm+=v;}
    for(int o=16;o>0;o>>=1)sm+=__shfl_down_sync(0xffffffff,sm,o);
    if(lid==0)smem[wid]=sm;__syncthreads();
    if(tid<32){float v=0;for(int w=0;w<nw;w++)v+=smem[w];for(int o=16;o>0;o>>=1)v+=__shfl_down_sync(0xffffffff,v,o);if(tid==0)smem[1]=v;}__syncthreads();
    sm=smem[1];float inv=1.f/(sm+1e-8f);
    for(int i=tid;i<cols;i+=blockDim.x)scores[row*cols+i]*=inv;
}

// Stage 3: S@V — one block per (token, Q-head)
__global__ void attn_sv_kernel(
    float* out, const float* scores, const float* v,
    int TQ,int Hq,int TK,int Hkv,int D,int groups)
{
    int t=blockIdx.x,h=blockIdx.y,kvh=h/groups,tid=threadIdx.x;
    for(int d=tid;d<D;d+=blockDim.x){
        float acc=0;
        for(int s=0;s<TK;s++)
            acc+=scores[(t*Hq+h)*TK+s]*v[(s*Hkv+kvh)*D+d];
        out[(t*Hq+h)*D+d]=acc;
    }
}

// GPU attention wrapper — allocates intermediate buffer, runs 3 kernels
static Tensor attn_gpu(const Tensor& q,const Tensor& k,const Tensor& v,
    int TQ,int Hq,int TK,int Hkv,int D,int groups,float scale)
{
    Tensor scores({TQ*Hq,TK},DType::F32,Device::CUDA);
    dim3 g1(TQ,Hq),b1(std::min(256,D));
    attn_qk_kernel<<<g1,b1>>>(scores.data<float>(),q.data<float>(),k.data<float>(),
        TQ,Hq,TK,Hkv,D,groups,scale);
    int rows=TQ*Hq,blk=std::min(512,((TK+31)/32)*32);if(blk<32)blk=32;
    attn_softmax_kernel<<<rows,blk>>>(scores.data<float>(),rows,TK);
    Tensor out({TQ,Hq,D},DType::F32,Device::CUDA);
    dim3 g3(TQ,Hq);
    attn_sv_kernel<<<g3,b1>>>(out.data<float>(),scores.data<float>(),v.data<float>(),
        TQ,Hq,TK,Hkv,D,groups);
    return std::move(out);
}
static void cuda_check(const char* ctx){cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)fprintf(stderr,"CUDA %s: %s\n",ctx,cudaGetErrorString(e));}

// ---- main ----
int main(){
    const char* model_dir="models/qwen2.5-0.5b";char buf[512];

    // Parse config
    snprintf(buf,sizeof(buf),"%s/config.json",model_dir);
    auto cfg=parse_config(buf);
    int D=cfg.hidden_size, Hq=cfg.num_attention_heads, Hkv=cfg.num_key_value_heads;
    int HD=cfg.head_dim, n_layers=cfg.num_hidden_layers, vocab=cfg.vocab_size;
    int groups=Hq/Hkv;float attn_scale=1.f/sqrtf((float)HD);

    printf("=== LightLLM Generation ===\nModel: %s, %dM params, %d layers\nD=%d Hq=%d Hkv=%d HD=%d vocab=%d\n",
        cfg.architecture.c_str(),(int)(cfg.total_params()/1e6),n_layers,D,Hq,Hkv,HD,vocab);

    // Load safetensors
    snprintf(buf,sizeof(buf),"%s/model.safetensors",model_dir);
    SafetensorsLoader loader(buf);

    // Load all weights as FP32 on GPU
    auto embed=load_f32(loader,"model.embed_tokens.weight");       // [vocab,D]
    auto fnorm=load_f32(loader,"model.norm.weight");               // [D]
    struct LW{Tensor q,k,v,o,a_n,gate,up,down,m_n;};
    std::vector<std::unique_ptr<LW>> layers;
    for(int i=0;i<n_layers;i++){
        auto ns=std::to_string(i);
        layers.push_back(std::make_unique<LW>(LW{
            load_f32(loader,"model.layers."+ns+".self_attn.q_proj.weight"),
            load_f32(loader,"model.layers."+ns+".self_attn.k_proj.weight"),
            load_f32(loader,"model.layers."+ns+".self_attn.v_proj.weight"),
            load_f32(loader,"model.layers."+ns+".self_attn.o_proj.weight"),
            load_f32(loader,"model.layers."+ns+".input_layernorm.weight"),
            load_f32(loader,"model.layers."+ns+".mlp.gate_proj.weight"),
            load_f32(loader,"model.layers."+ns+".mlp.up_proj.weight"),
            load_f32(loader,"model.layers."+ns+".mlp.down_proj.weight"),
            load_f32(loader,"model.layers."+ns+".post_attention_layernorm.weight"),
        }));
    }
    printf("All %d layers loaded (FP32 GPU)\n",n_layers);

    // Prompt tokens: "The capital of France is"
    std::vector<int> prompt={576,8319,315,13466,374};
    int max_new=12, eos_id=151643, max_seq=(int)prompt.size()+max_new;

    // Allocate contiguous KV cache per layer [max_seq, Hkv, hd]
    struct KVC{Tensor k,v;};
    std::vector<KVC> kv;
    for(int l=0;l<n_layers;l++)
        kv.push_back({Tensor({max_seq,Hkv,HD},DType::F32,Device::CUDA),
                       Tensor({max_seq,Hkv,HD},DType::F32,Device::CUDA)});

    auto t_start=std::chrono::steady_clock::now();

    // ==================== PREFILL ====================
    int P=(int)prompt.size();
    Tensor h_pf({P,D},DType::F32,Device::CUDA);

    // Embedding: copy rows from embed weight
    {
        auto ew_cpu=std::vector<float>(vocab*D);
        embed.copy_to(ew_cpu.data(),embed.nbytes());
        std::vector<float> emb(P*D);
        for(int t=0;t<P;t++)
            memcpy(&emb[t*D],&ew_cpu[prompt[t]*D],D*sizeof(float));
        h_pf.copy_from(emb.data(),emb.size()*sizeof(float));
    }

    // Prefill forward: process all P tokens through 24 layers
    for(int l=0;l<n_layers;l++){
        auto& L=*layers[l];auto& kvc=kv[l];

        // RMSNorm → Q/K/V projections
        auto nm=rms_norm(h_pf,L.a_n);
        auto q_g=gemm(nm,L.q,true), k_g=gemm(nm,L.k,true), v_g=gemm(nm,L.v,true);

        // Reshape to multi-head: [P,D] → [P,H,hd]
        std::vector<int> q_shape={P,Hq,HD}, kv_shape={P,Hkv,HD};
        auto q3=q_g.view(q_shape), k3=k_g.view(kv_shape), v3=v_g.view(kv_shape);

        // RoPE for prefill positions
        Tensor cos({P,HD/2},DType::F32,Device::CUDA), sin({P,HD/2},DType::F32,Device::CUDA);
        {
            size_t n=P*(HD/2);
            std::vector<float> vc(n), vs(n);
            for(int t=0;t<P;t++)for(int d=0;d<HD/2;d++){
                float theta=(float)t/powf(cfg.rope_theta,2.f*d/HD);
                vc[t*HD/2+d]=cosf(theta);vs[t*HD/2+d]=sinf(theta);
            }
            cos.copy_from(vc.data(),vc.size()*sizeof(float));
            sin.copy_from(vs.data(),vs.size()*sizeof(float));
        }
        rope(q3,&k3,cos,sin);cudaDeviceSynchronize();

        // Store K/V into cache
        size_t kv_bytes=P*Hkv*HD*sizeof(float);
        cudaMemcpy(kvc.k.raw(),k3.raw(),kv_bytes,cudaMemcpyDeviceToDevice);
        cudaMemcpy(kvc.v.raw(),v3.raw(),kv_bytes,cudaMemcpyDeviceToDevice);

        // GPU attention
        Tensor at=attn_gpu(q3,k3,v3,P,Hq,P,Hkv,HD,groups,attn_scale);
        at=at.view(std::vector<int>{P,D});                      // reshape to [P,D]
        at=gemm(at,L.o,true);                   // O projection
        h_pf=ops::add(h_pf,at);                        // residual

        // MLP block
        nm=rms_norm(h_pf,L.m_n);
        auto gate=gemm(nm,L.gate,true), up=gemm(nm,L.up,true);
        silu_inplace(gate);
        auto mlp_h=ops::mul(gate,up);
        mlp_h=gemm(mlp_h,L.down,true);
        h_pf=ops::add(h_pf,mlp_h);
    }
    // Final norm → get last token's hidden state
    h_pf=rms_norm(h_pf,fnorm);
    Tensor cur_h({1,D},DType::F32,Device::CUDA);
    {
        const float* src=(const float*)h_pf.raw()+(P-1)*D;
        cudaMemcpy(cur_h.raw(),src,D*sizeof(float),cudaMemcpyDeviceToDevice);
    }

    // ==================== DECODE ====================
    std::vector<int> generated=prompt;
    int cur_len=P;
    std::mt19937 rng(42);
    printf("Prefill done (%d tokens), starting decode...\n",P);

    for(int step=0;step<max_new&&cur_len<max_seq;step++){
        Tensor h_dec({1,D},DType::F32,Device::CUDA);
        cudaMemcpy(h_dec.raw(),cur_h.raw(),D*sizeof(float),cudaMemcpyDeviceToDevice);

        for(int l=0;l<n_layers;l++){
            auto& L=*layers[l];auto& kvc=kv[l];

            auto nm=rms_norm(h_dec,L.a_n);
            auto q_g=gemm(nm,L.q,true),k_g=gemm(nm,L.k,true),v_g=gemm(nm,L.v,true);
            auto q3=q_g.view(std::vector<int>{1,Hq,HD}),k3=k_g.view(std::vector<int>{1,Hkv,HD}),v3=v_g.view(std::vector<int>{1,Hkv,HD});

            // RoPE for current position
            Tensor cos({1,HD/2},DType::F32,Device::CUDA),sin({1,HD/2},DType::F32,Device::CUDA);
            {
                std::vector<float> vc(HD/2),vs(HD/2);
                for(int d=0;d<HD/2;d++){
                    float theta=(float)cur_len/powf(cfg.rope_theta,2.f*d/HD);
                    vc[d]=cosf(theta);vs[d]=sinf(theta);
                }
                cos.copy_from(vc.data(),vc.size()*sizeof(float));
                sin.copy_from(vs.data(),vs.size()*sizeof(float));
            }
            rope(q3,&k3,cos,sin);cudaDeviceSynchronize();

            // Append new K/V to cache
            float* k_dst=(float*)kvc.k.raw()+cur_len*Hkv*HD;
            float* v_dst=(float*)kvc.v.raw()+cur_len*Hkv*HD;
            size_t tkv_bytes=Hkv*HD*sizeof(float);
            cudaMemcpy(k_dst,k3.raw(),tkv_bytes,cudaMemcpyDeviceToDevice);
            cudaMemcpy(v_dst,v3.raw(),tkv_bytes,cudaMemcpyDeviceToDevice);

            // GPU attention: 1 query vs all history
            Tensor at=attn_gpu(q3,kvc.k,kvc.v,1,Hq,cur_len+1,Hkv,HD,groups,attn_scale);
            at=at.view(std::vector<int>{1,D});
            at=gemm(at,L.o,true);h_dec=ops::add(h_dec,at);

            // MLP
            nm=rms_norm(h_dec,L.m_n);
            auto gate=gemm(nm,L.gate,true),up=gemm(nm,L.up,true);
            silu_inplace(gate);
            auto mlp_h=ops::mul(gate,up);
            mlp_h=gemm(mlp_h,L.down,true);
            h_dec=ops::add(h_dec,mlp_h);
        }
        h_dec=rms_norm(h_dec,fnorm);

        // GPU lm_head — full logits computed on device, only result downloaded
        auto logits_gpu = ops::lm_head_logits(h_dec, embed);
        std::vector<float> logits(vocab);
        logits_gpu.copy_to(logits.data(), vocab * sizeof(float));

        ops::SamplingParams sp;
        sp.temperature = 0.8f;
        sp.top_k = 40;
        sp.top_p = 0.9f;
        sp.min_p = 0.0f;
        sp.seed = 42 + step;
        int next = ops::sample(logits.data(), vocab, sp, rng);

        generated.push_back(next);
        cur_len++;
        cur_h=std::move(h_dec);

        printf("  step %d: token=%d\n",step,next);
        if(next==eos_id)break;
    }

    auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now()-t_start).count();
    int new_tokens=(int)generated.size()-(int)prompt.size();

    printf("\n=== Result ===\n");
    printf("Prompt: ");for(int id:prompt)printf("%d ",id);printf("\n");
    printf("Output: ");for(int id:generated)printf("%d ",id);printf("\n");
    printf("Generated %d tokens in %.1f s (%.1f tok/s)\n",
        new_tokens,elapsed/1000.0,new_tokens*1000.0/elapsed);
    return 0;
}
