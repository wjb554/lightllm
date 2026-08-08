/// Continuous Batching Demo — batched GEMM + per-request attention.
/// Multiple users, batched forward pass.
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
#include "lightllm/ops/lm_head.h"
#include "lightllm/ops/sampling.h"
#include <random>

using namespace lightllm;
using namespace lightllm::model;
using namespace lightllm::ops;

static std::vector<float> bf16f32(const std::vector<char>& r){std::vector<float> f(r.size()/2);for(size_t i=0;i<f.size();i++){uint16_t b=((uint16_t*)r.data())[i];uint32_t u=b<<16;f[i]=*(float*)&u;}return f;}
static Tensor load_f32(SafetensorsLoader& l,const std::string& n){auto cpu=l.load_tensor(n);std::vector<char> raw(cpu.nbytes());cpu.copy_to(raw.data(),cpu.nbytes());auto f32=bf16f32(raw);Tensor gpu(l.get_info(n)->shape,DType::F32,Device::CUDA);gpu.copy_from(f32.data(),f32.size()*sizeof(float));return gpu;}

// GPU attention kernels (3-stage)
__global__ void ak_qk(float* s,const float* q,const float* k,int TQ,int Hq,int TK,int Hkv,int D,int g,float sc){int t=blockIdx.x,h=blockIdx.y,kvh=h/g,tid=threadIdx.x;for(int i=tid;i<TK;i+=blockDim.x){float d=0;for(int dd=0;dd<D;dd++)d+=q[(t*Hq+h)*D+dd]*k[(i*Hkv+kvh)*D+dd];s[(t*Hq+h)*TK+i]=d*sc;}}
__global__ void ak_sm(float* s,int R,int C){int r=blockIdx.x,t=threadIdx.x;__shared__ float m[64];int w=t/32,l=t%32,nw=blockDim.x/32;float mx=-INFINITY;for(int i=t;i<C;i+=blockDim.x)mx=fmaxf(mx,s[r*C+i]);for(int o=16;o>0;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));if(l==0)m[w]=mx;__syncthreads();if(t<32){float v=(t<nw)?m[t]:-INFINITY;for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,o));if(t==0)m[0]=v;}__syncthreads();mx=m[0];float sm=0;for(int i=t;i<C;i+=blockDim.x){float v=expf(s[r*C+i]-mx);s[r*C+i]=v;sm+=v;}for(int o=16;o>0;o>>=1)sm+=__shfl_down_sync(0xffffffff,sm,o);if(l==0)m[w]=sm;__syncthreads();if(t<32){float v=0;for(int i=0;i<nw;i++)v+=m[i];for(int o=16;o>0;o>>=1)v+=__shfl_down_sync(0xffffffff,v,o);if(t==0)m[1]=v;}__syncthreads();float inv=1.f/(m[1]+1e-8f);for(int i=t;i<C;i+=blockDim.x)s[r*C+i]*=inv;}
__global__ void ak_sv(float* o,const float* s,const float* v,int TQ,int Hq,int TK,int Hkv,int D,int g){int t=blockIdx.x,h=blockIdx.y,kvh=h/g,tid=threadIdx.x;for(int d=tid;d<D;d+=blockDim.x){float a=0;for(int i=0;i<TK;i++)a+=s[(t*Hq+h)*TK+i]*v[(i*Hkv+kvh)*D+d];o[(t*Hq+h)*D+d]=a;}}
static Tensor gpu_attn(const Tensor& q,const Tensor& k,const Tensor& v,int TQ,int Hq,int TK,int Hkv,int D,int groups,float sc){Tensor s({TQ*Hq,TK},DType::F32,Device::CUDA);dim3 g1(TQ,Hq),b1(std::min(256,D));ak_qk<<<g1,b1>>>(s.data<float>(),q.data<float>(),k.data<float>(),TQ,Hq,TK,Hkv,D,groups,sc);int R=TQ*Hq,blk=std::min(512,((TK+31)/32)*32);if(blk<32)blk=32;ak_sm<<<R,blk>>>(s.data<float>(),R,TK);Tensor o({TQ,Hq,D},DType::F32,Device::CUDA);dim3 g3(TQ,Hq);ak_sv<<<g3,b1>>>(o.data<float>(),s.data<float>(),v.data<float>(),TQ,Hq,TK,Hkv,D,groups);return std::move(o);}

// ========================================================================
struct User{
    std::vector<int> prompt, generated;
    int max_new, cur_len;bool active;
    struct KVC{Tensor k,v;KVC()=default;KVC(Tensor kk,Tensor vv):k(std::move(kk)),v(std::move(vv)){}};std::vector<KVC> kv;
};

int main(){
    const char* mdl="models/qwen2.5-0.5b";char buf[512];
    snprintf(buf,sizeof(buf),"%s/config.json",mdl);auto cfg=parse_config(buf);
    snprintf(buf,sizeof(buf),"%s/model.safetensors",mdl);SafetensorsLoader loader(buf);
    int D=cfg.hidden_size,Hq=cfg.num_attention_heads,Hkv=cfg.num_key_value_heads;
    int HD=cfg.head_dim,NL=cfg.num_hidden_layers,V=cfg.vocab_size;
    int groups=Hq/Hkv;float attn_scale=1.f/sqrtf((float)HD);
    printf("=== Continuous Batching ===\n%d users, %dM model\n\n",3,(int)(cfg.total_params()/1e6));

    // Load model
    auto embed=load_f32(loader,"model.embed_tokens.weight");
    auto fnorm=load_f32(loader,"model.norm.weight");
    struct LW{Tensor q,k,v,o,a_n,gate,up,down,m_n;};
    std::vector<std::unique_ptr<LW>> layers;
    for(int i=0;i<NL;i++){auto ns=std::to_string(i);layers.push_back(std::make_unique<LW>(LW{load_f32(loader,"model.layers."+ns+".self_attn.q_proj.weight"),load_f32(loader,"model.layers."+ns+".self_attn.k_proj.weight"),load_f32(loader,"model.layers."+ns+".self_attn.v_proj.weight"),load_f32(loader,"model.layers."+ns+".self_attn.o_proj.weight"),load_f32(loader,"model.layers."+ns+".input_layernorm.weight"),load_f32(loader,"model.layers."+ns+".mlp.gate_proj.weight"),load_f32(loader,"model.layers."+ns+".mlp.up_proj.weight"),load_f32(loader,"model.layers."+ns+".mlp.down_proj.weight"),load_f32(loader,"model.layers."+ns+".post_attention_layernorm.weight")}));}
    auto ew_cpu=std::vector<float>(V*D);embed.copy_to(ew_cpu.data(),embed.nbytes());
    printf("Model loaded\n");

    // 3 users with different prompts
    int max_seq=48;
    std::vector<User> users(3);
    users[0].prompt={576,8319,315,13466,374}; // "The capital of France is"
    users[1].prompt={15837,467,315,1605};     // "The weather today is"
    users[2].prompt={10585,374,279,18865};    // "I think therefore I"
    for(auto& u:users){u.max_new=16;u.active=true;u.cur_len=0;}

    // Prefill all users
    for(auto& u:users){
        int P=(int)u.prompt.size();u.cur_len=P;u.generated=u.prompt;
        for(int l=0;l<NL;l++)u.kv.push_back({Tensor({max_seq,Hkv,HD},DType::F32,Device::CUDA),Tensor({max_seq,Hkv,HD},DType::F32,Device::CUDA)});
        Tensor h_pf({P,D},DType::F32,Device::CUDA);
        {std::vector<float> emb(P*D);for(int t=0;t<P;t++)memcpy(&emb[t*D],&ew_cpu[u.prompt[t]*D],D*sizeof(float));h_pf.copy_from(emb.data(),emb.size()*sizeof(float));}
        for(int l=0;l<NL;l++){auto&L=*layers[l];auto&kv=u.kv[l];
            auto nm=rms_norm(h_pf,L.a_n);auto qg=gemm(nm,L.q,true),kg=gemm(nm,L.k,true),vg=gemm(nm,L.v,true);
            auto q3=qg.view(std::vector<int>{P,Hq,HD}),k3=kg.view(std::vector<int>{P,Hkv,HD}),v3=vg.view(std::vector<int>{P,Hkv,HD});
            Tensor cos({P,HD/2},DType::F32,Device::CUDA),sin({P,HD/2},DType::F32,Device::CUDA);
            {std::vector<float>vc(P*HD/2),vs(P*HD/2);for(int t=0;t<P;t++)for(int d=0;d<HD/2;d++){float th=(float)t/powf(cfg.rope_theta,2.f*d/HD);vc[t*HD/2+d]=cosf(th);vs[t*HD/2+d]=sinf(th);}cos.copy_from(vc.data(),vc.size()*sizeof(float));sin.copy_from(vs.data(),vs.size()*sizeof(float));}
            rope(q3,&k3,cos,sin);cudaDeviceSynchronize();
            cudaMemcpy(kv.k.raw(),k3.raw(),P*Hkv*HD*sizeof(float),cudaMemcpyDeviceToDevice);
            cudaMemcpy(kv.v.raw(),v3.raw(),P*Hkv*HD*sizeof(float),cudaMemcpyDeviceToDevice);
            Tensor at=gpu_attn(q3,k3,v3,P,Hq,P,Hkv,HD,groups,attn_scale);
            at=at.view(std::vector<int>{P,D});at=gemm(at,L.o,true);h_pf=ops::add(h_pf,at);
            nm=rms_norm(h_pf,L.m_n);auto gate=gemm(nm,L.gate,true),up=gemm(nm,L.up,true);silu_inplace(gate);auto mlp=ops::mul(gate,up);mlp=gemm(mlp,L.down,true);h_pf=ops::add(h_pf,mlp);
        }
    }
    printf("All users prefilled\n\n");

    // ===== Batched Decode Loop =====
    auto t0=std::chrono::steady_clock::now();
    int total_tokens=0;

    while(true){
        // Collect active users
        std::vector<int> active;for(int i=0;i<(int)users.size();i++)if(users[i].active)active.push_back(i);
        if(active.empty())break;
        int BS=(int)active.size();

        // Build batch hidden states: [BS, D]
        Tensor h_batch({BS,D},DType::F32,Device::CUDA);
        {
            std::vector<float> batch_data(BS*D);
            for(int bi=0;bi<BS;bi++){
                auto& u=users[active[bi]];
                int tok=u.generated.back();
                memcpy(&batch_data[bi*D],&ew_cpu[tok*D],D*sizeof(float));
            }
            h_batch.copy_from(batch_data.data(),batch_data.size()*sizeof(float));
        }

        // ---- Batch Forward: ALL GEMMs process [BS, D] at once! ----
        Tensor h(std::move(h_batch));
        for(int l=0;l<NL;l++){
            auto& L=*layers[l];

            // RMSNorm + Q/K/V projections — BATCHED [BS, D]
            auto nm=rms_norm(h,L.a_n);
            auto q_g=gemm(nm,L.q,true),k_g=gemm(nm,L.k,true),v_g=gemm(nm,L.v,true);

            // ---- Per-request attention (sequential within batch) ----
            // Allocate output buffer [BS, D]
            Tensor ao({BS,D},DType::F32,Device::CUDA);
            for(int bi=0;bi<BS;bi++){
                int ui=active[bi];auto& u=users[ui];
                // Extract this request's Q/K/V from batch
                Tensor q1({1,Hq,HD},DType::F32,Device::CUDA),k1({1,Hkv,HD},DType::F32,Device::CUDA),v1({1,Hkv,HD},DType::F32,Device::CUDA);
                // Copy from batch: q_g[bi, :] → q1[0, :]
                cudaMemcpy(q1.raw(),(float*)q_g.raw()+bi*Hq*HD, Hq*HD*sizeof(float),cudaMemcpyDeviceToDevice);
                cudaMemcpy(k1.raw(),(float*)k_g.raw()+bi*Hkv*HD, Hkv*HD*sizeof(float),cudaMemcpyDeviceToDevice);
                cudaMemcpy(v1.raw(),(float*)v_g.raw()+bi*Hkv*HD, Hkv*HD*sizeof(float),cudaMemcpyDeviceToDevice);

                // RoPE for this request's current position
                int pos=u.cur_len;
                Tensor cos({1,HD/2},DType::F32,Device::CUDA),sin({1,HD/2},DType::F32,Device::CUDA);
                {std::vector<float>vc(HD/2),vs(HD/2);for(int d=0;d<HD/2;d++){float th=(float)pos/powf(cfg.rope_theta,2.f*d/HD);vc[d]=cosf(th);vs[d]=sinf(th);}cos.copy_from(vc.data(),vc.size()*sizeof(float));sin.copy_from(vs.data(),vs.size()*sizeof(float));}
                rope(q1,&k1,cos,sin);cudaDeviceSynchronize();

                // Store K/V
                auto& kv=u.kv[l];
                cudaMemcpy((float*)kv.k.raw()+pos*Hkv*HD,k1.raw(),Hkv*HD*sizeof(float),cudaMemcpyDeviceToDevice);
                cudaMemcpy((float*)kv.v.raw()+pos*Hkv*HD,v1.raw(),Hkv*HD*sizeof(float),cudaMemcpyDeviceToDevice);

                // GPU Attention: this request's Q vs its full KV history
                Tensor at=gpu_attn(q1,kv.k,kv.v,1,Hq,pos+1,Hkv,HD,groups,attn_scale);
                at=at.view(std::vector<int>{1,D});
                // Copy attention output to batch output buffer
                cudaMemcpy((float*)ao.raw()+bi*D,at.raw(),D*sizeof(float),cudaMemcpyDeviceToDevice);
            }

            // O projection — BATCHED [BS, D]
            ao=gemm(ao,L.o,true);
            h=ops::add(h,ao);

            // MLP — BATCHED
            nm=rms_norm(h,L.m_n);
            auto gate=gemm(nm,L.gate,true),up=gemm(nm,L.up,true);
            silu_inplace(gate);
            auto mlp=ops::mul(gate,up);
            mlp=gemm(mlp,L.down,true);
            h=ops::add(h,mlp);
        }
        h=rms_norm(h,fnorm);

        // Per-token GPU lm_head + sampling
        std::mt19937 rng(42);
        for(int bi=0;bi<BS;bi++){
            int ui=active[bi];auto& u=users[ui];
            // Extract hidden state for this user: h[bi, :] → [D]
            Tensor h1({D},DType::F32,Device::CUDA);
            cudaMemcpy(h1.raw(),(float*)h.raw()+bi*D,D*sizeof(float),cudaMemcpyDeviceToDevice);
            // GPU lm_head: [D] × embed[vocab,D]^T → logits[vocab]
            auto logits_gpu = lm_head_logits(h1, embed);
            std::vector<float> logits(V);
            logits_gpu.copy_to(logits.data(), logits_gpu.nbytes());
            // Sampling
            ops::SamplingParams sp;
            sp.temperature = 0.8f;
            sp.top_k = 40;
            sp.top_p = 0.9f;
            sp.min_p = 0.0f;
            sp.seed = 42 + u.cur_len;
            int next = ops::sample(logits.data(), V, sp, rng);
            u.generated.push_back(next);u.cur_len++;total_tokens++;
            if(next==151643||u.cur_len>=max_seq)u.active=false;
        }

        // Print progress
        int still_active=0;for(auto& u:users)if(u.active)still_active++;
        printf("  step: batch=%d active=%d tokens=%d\n",BS,still_active,total_tokens);
    }

    auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count();

    printf("\n=== Results ===\n");
    for(int i=0;i<(int)users.size();i++){
        auto& u=users[i];
        printf("User %d: %zu→%zu tokens\n",i,u.prompt.size(),u.generated.size());
        printf("  IDs: ");for(int id:u.generated)printf("%d ",id);printf("\n");
    }
    printf("\n%d tokens across %zu users in %.1fs (%.1f tok/s batched)\n",total_tokens,users.size(),ms/1000.0,total_tokens*1000.0/ms);
    return 0;
}
