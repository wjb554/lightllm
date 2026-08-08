/// LightLLM — fp16 mixed-precision quick test.
/// Usage: bench_fp16
#include <cstdio>
#include <cstring>
#include <chrono>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "lightllm/tensor.h"
#include "lightllm/model/model_loader.h"
#include "lightllm/model/model_config.h"
#include "lightllm/ops/gemm.h"

using namespace lightllm;
using namespace lightllm::model;
using namespace lightllm::ops;

// Load BF16→fp16 on GPU
static Tensor load_f16(SafetensorsLoader& l, const std::string& n) {
    Tensor cpu = l.load_tensor(n);
    auto* info = l.get_info(n);
    std::vector<char> raw(cpu.nbytes()); cpu.copy_to(raw.data(), cpu.nbytes());
    size_t nel = raw.size()/2;
    std::vector<half> f16(nel);
    for(size_t i=0;i<nel;i++){
        uint32_t bits=((uint32_t)((const uint16_t*)raw.data())[i])<<16;
        f16[i]=__float2half(*(float*)&bits);
    }
    Tensor gpu(info->shape, DType::F16, Device::CUDA);
    gpu.copy_from(f16.data(), f16.size()*sizeof(half));
    return gpu;
}

int main() {
    printf("=== LightLLM fp16 Memory Benchmark ===\n");
    fflush(stdout);

    const char* mdl = "models/qwen2.5-0.5b";
    char buf[512];

    snprintf(buf,sizeof(buf),"%s/config.json",mdl);
    auto cfg = parse_config(buf);
    int D=cfg.hidden_size, NL=cfg.num_hidden_layers;
    printf("Config: D=%d layers=%d\n", D, NL);
    fflush(stdout);

    snprintf(buf,sizeof(buf),"%s/model.safetensors",mdl);
    printf("Opening: %s\n", buf);
    fflush(stdout);
    SafetensorsLoader loader(buf);

    printf("Loading Qwen2.5-0.5B as fp16...\n");
    fflush(stdout);

    // Load all weights as fp16
    printf("Loading embed_tokens...\n"); fflush(stdout);
    auto embed = load_f16(loader,"model.embed_tokens.weight");
    printf("  embed: [%d,%d] %.0f MB\n", embed.size(0), embed.size(1), embed.nbytes()/1048576.0);
    fflush(stdout);

    struct LW{Tensor q,k,v,o,a_n,gate,up,down,m_n;};
    std::vector<std::unique_ptr<LW>> layers;
    for(int i=0;i<NL;i++){
        auto ns=std::to_string(i);
        printf("  layer %d...\n", i); fflush(stdout);
        layers.push_back(std::make_unique<LW>(LW{
            load_f16(loader,"model.layers."+ns+".self_attn.q_proj.weight"),
            load_f16(loader,"model.layers."+ns+".self_attn.k_proj.weight"),
            load_f16(loader,"model.layers."+ns+".self_attn.v_proj.weight"),
            load_f16(loader,"model.layers."+ns+".self_attn.o_proj.weight"),
            load_f16(loader,"model.layers."+ns+".input_layernorm.weight"),
            load_f16(loader,"model.layers."+ns+".mlp.gate_proj.weight"),
            load_f16(loader,"model.layers."+ns+".mlp.up_proj.weight"),
            load_f16(loader,"model.layers."+ns+".mlp.down_proj.weight"),
            load_f16(loader,"model.layers."+ns+".post_attention_layernorm.weight"),
        }));
    }
    printf("Loaded %d layers\n",NL); fflush(stdout);

    // Calculate memory
    size_t total=0;
    total+=embed.nbytes();
    for(auto& lw:layers){
        total+=lw->q.nbytes()+lw->k.nbytes()+lw->v.nbytes()+lw->o.nbytes()
              +lw->a_n.nbytes()+lw->gate.nbytes()+lw->up.nbytes()+lw->down.nbytes()
              +lw->m_n.nbytes();
    }
    printf("\n=== Memory Usage ===\n");
    printf("Model weights (fp16): %.0f MB\n", total/1048576.0);
    printf("Model weights (fp32): %.0f MB\n", total*2.0/1048576.0);
    printf("Saved: %.0f MB (50%%)\n", total/1048576.0);

    // Quick GEMM test: 1 forward layer
    printf("\n=== Quick GEMM Test (fp16->fp32) ===\n"); fflush(stdout);
    printf("Creating fp16 input...\n"); fflush(stdout);
    Tensor h_f16({1,D}, DType::F16, Device::CUDA);
    {
        std::vector<half> ones(D, __float2half(1.0f));
        h_f16.copy_from(ones.data(), D*sizeof(half));
    }
    printf("Input created, running GEMM...\n"); fflush(stdout);

    auto t0=std::chrono::steady_clock::now();
    auto q_out=gemm_f16f32(h_f16, layers[0]->q, true);
    printf("GEMM done, sync...\n"); fflush(stdout);
    cudaDeviceSynchronize();
    auto us=std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-t0).count();

    size_t flops=2ULL*D*D;
    printf("Q-proj [1,%d]x[%d,%d]: %lld us (%.0f GFLOPS)\n",
           D,D,D,us,flops/(us*1e3));
    float first_val;
    q_out.copy_to(&first_val, sizeof(float));
    printf("Output[0]: %.3f\n", first_val); fflush(stdout);

    printf("\n=== OK ===\n");
    return 0;
}
