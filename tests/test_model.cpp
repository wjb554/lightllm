/// Phase 3 test: safetensors loader + config parser
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

#include "lightllm/tensor.h"
#include "lightllm/model/model_loader.h"
#include "lightllm/model/model_config.h"

using namespace lightllm;
using namespace lightllm::model;

static int passed=0,failed=0;
static void CHECK(const char* n,bool c){if(c)passed++;else{failed++;fprintf(stderr,"  FAIL: %s\n",n);}}

int main() {
    const char* model_dir = "models/qwen2.5-0.5b";
    char path_buf[512];

    // 1. Parse config
    snprintf(path_buf,sizeof(path_buf),"%s/config.json",model_dir);
    printf("=== Config Parser ===\n");
    auto cfg = parse_config(path_buf);
    printf("%s\n",cfg.to_string().c_str());

    CHECK("arch",cfg.architecture=="Qwen2ForCausalLM");
    CHECK("hidden",cfg.hidden_size==896);
    CHECK("inter",cfg.intermediate_size==4864);
    CHECK("heads",cfg.num_attention_heads==14);
    CHECK("kv_heads",cfg.num_key_value_heads==2);
    CHECK("layers",cfg.num_hidden_layers==24);
    CHECK("head_dim",cfg.head_dim==64);
    CHECK("vocab",cfg.vocab_size==151936);
    CHECK("tied",cfg.tie_word_embeddings==true);
    CHECK("params",cfg.total_params()>400000000 && cfg.total_params()<500000000);

    // 2. Load safetensors
    snprintf(path_buf,sizeof(path_buf),"%s/model.safetensors",model_dir);
    printf("\n=== Safetensors Loader ===\n");
    SafetensorsLoader loader(path_buf);
    auto names = loader.tensor_names();
    printf("Loaded %zu tensors\n",names.size());
    CHECK("num_tensors",names.size()>=25*3+2); // 24 layers + embed + norm

    // Verify critical tensors exist
    CHECK("has_embed",loader.get_info("model.embed_tokens.weight")!=nullptr);
    CHECK("has_q0",loader.get_info("model.layers.0.self_attn.q_proj.weight")!=nullptr);
    CHECK("has_k0",loader.get_info("model.layers.0.self_attn.k_proj.weight")!=nullptr);
    CHECK("has_v0",loader.get_info("model.layers.0.self_attn.v_proj.weight")!=nullptr);
    CHECK("has_o0",loader.get_info("model.layers.0.self_attn.o_proj.weight")!=nullptr);
    CHECK("has_gate0",loader.get_info("model.layers.0.mlp.gate_proj.weight")!=nullptr);
    CHECK("has_up0",loader.get_info("model.layers.0.mlp.up_proj.weight")!=nullptr);
    CHECK("has_down0",loader.get_info("model.layers.0.mlp.down_proj.weight")!=nullptr);
    CHECK("has_norm",loader.get_info("model.norm.weight")!=nullptr);

    // Verify shape correctness for key tensors
    auto* embed_info = loader.get_info("model.embed_tokens.weight");
    CHECK("embed_shape",embed_info->shape.size()==2 &&
          embed_info->shape[0]==151936 && embed_info->shape[1]==896);
    CHECK("embed_dtype",embed_info->dtype==DType::BF16);

    auto* q_info = loader.get_info("model.layers.0.self_attn.q_proj.weight");
    CHECK("q_shape",q_info->shape.size()==2 &&
          q_info->shape[0]==896 && q_info->shape[1]==896);

    auto* k_info = loader.get_info("model.layers.0.self_attn.k_proj.weight");
    CHECK("k_shape",k_info->shape.size()==2 &&
          k_info->shape[0]==128 && k_info->shape[1]==896);
    // K dim = num_kv_heads * head_dim = 2 * 64 = 128

    auto* down_info = loader.get_info("model.layers.0.mlp.down_proj.weight");
    CHECK("down_shape",down_info->shape.size()==2 &&
          down_info->shape[0]==896 && down_info->shape[1]==4864);

    // 3. Load a tensor to GPU and verify byte-level correctness
    printf("\n=== GPU Loading ===\n");
    auto* tn_info = loader.get_info("model.norm.weight"); // [896] BF16
    Tensor cpu_t = loader.load_tensor("model.norm.weight"); // loads to CPU
    CHECK("norm_cpu_shape",cpu_t.shape()==std::vector<int>{896});
    CHECK("norm_cpu_dtype",cpu_t.dtype()==DType::BF16);

    // Load to GPU
    Tensor gpu_t({896},DType::BF16,Device::CUDA);
    loader.load_into("model.norm.weight",gpu_t);
    CHECK("norm_gpu_shape",gpu_t.shape()==std::vector<int>{896});

    // Copy GPU back and compare with CPU load
    std::vector<char> cpu_bytes(cpu_t.nbytes()), gpu_bytes(gpu_t.nbytes());
    cpu_t.copy_to(cpu_bytes.data(),cpu_t.nbytes());
    gpu_t.copy_to(gpu_bytes.data(),gpu_t.nbytes());
    CHECK("byte_identical",memcmp(cpu_bytes.data(),gpu_bytes.data(),cpu_t.nbytes())==0);

    printf("\n%s: %d passed, %d failed\n",failed?"SOME FAILURES":"ALL PASSED",passed,failed);
    return failed?1:0;
}
