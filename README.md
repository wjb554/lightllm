# LightLLM

从零构建的 C++/CUDA LLM 推理引擎，对标 vLLM/SGLang。

**95 源文件 · ~23,000 行代码 · 19 测试 · ~202,000 断言 · 0 失败**

## 特性

- **12 类手写 CUDA kernel**：GEMM (5 级后端)、PagedAttention、RMSNorm、RoPE、Softmax 等
- **PagedAttention**：Float4 向量化 + 双缓冲 + Split-K 三优化，Paged Prefill Attention
- **前缀缓存**：Hash 前缀缓存（vLLM 风格）+ Radix Tree 前缀缓存（SGLang 风格），可配置切换
- **Continuous Batching**：DecodeFirst/PrefillFirst/FCFS 调度，混合 prefill+decode 批量处理
- **采样系统**：temperature、top-k、top-p、min-p
- **fp16 混合精度**：权重 fp16 + Tensor Core GEMM，显存减半

## 性能 (Qwen2.5-0.5B fp32)

### 本机 RTX 2060 6GB (Windows, CUDA 13.2)

| 场景 | 吞吐 |
|------|------|
| 单用户 | 15.8–18.7 tok/s |
| 3 用户 Continuous Batching | 25.1–25.3 tok/s |
| 32 并发 | 稳定 |
| GEMM fp32 手写 (Vectorized) | 1646 GFLOPS（cuBLAS 2833 的 58%） |
| GEMM fp32 手写 (DoubleBuf) | 1502 GFLOPS（K=1024 时 swap 开销 > 双缓冲收益） |
| fp16 GEMM (Tensor Core) | ~4.7 TFLOPS |

### 云端 RTX 4090 24GB (Linux, CUDA 11.8)

| 场景 | 吞吐 | 较 2060 |
|------|------|------|
| 单用户 | 22.9 tok/s | 1.2–1.5× |
| 3 用户 Continuous Batching | **103.7 tok/s** | **4.1×** |
| 16 用户混合长度 | ~40 tok/s | 2× |
| P50 TTFT / P50 TPOT | 879 / 385 ms | 3.6× / 4.0× |
| 并发上限（修复悬垂指针后） | 512 稳定 | 32→512 (16×) |

详细数据见 `docs/REPORT_rtx2060.md` 与 `docs/REPORT_rtx4090.md`。

## 环境要求

| 组件 | 版本 |
|------|------|
| Windows 10/11 或 Linux | — |
| CUDA Toolkit | 12.0+ (开发用 13.2) |
| MSVC (Windows) | 2022+ |
| GCC (Linux) | 11+ |
| CMake | 3.20+ |
| Ninja | 任意版本 |

## 快速开始

### 1. 下载模型

```bash
# 下载 Qwen2.5-0.5B 到 models/qwen2.5-0.5b/
# 需要三个文件: config.json, model.safetensors, tokenizer.json
# HuggingFace: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
mkdir -p models/qwen2.5-0.5b
# 将下载的文件放入该目录
```

### 2. 构建

```bash
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -j4
```

### 3. 运行测试

```bash
cd ..  # 回到项目根目录
./build/test_generate.exe     # 单用户生成演示
./build/test_concurrent.exe   # 3 用户 Continuous Batching
./build/test_sampling.exe     # 采样测试 (140k 断言)
./build/test_paged_attention.exe  # PagedAttention 测试
```

### 4. 运行基准测试

```bash
./build/bench_concurrency.exe    # 不同并发数吞吐对比
./build/bench_throughput.exe     # 高负载混合长度仿真
./build/bench_fp16.exe           # fp16 显存占用测试
```

### 5. VS Code 开发

项目包含 `.vscode/` 配置：
- `Ctrl+Shift+B` 编译
- `F5` 调试
- 推荐安装：C/C++ (Microsoft)、CMake Tools

## 项目结构

```
lightllm/
├── include/lightllm/
│   ├── tensor.h              # GPU Tensor 抽象
│   ├── ops/                  # 算子层 (gemm, norm, rope, attention, sampling, ...)
│   ├── kv_cache/             # PagedAttention + BlockAllocator + PrefixCache
│   ├── engine/               # Scheduler + EngineServer + BatchMainLoop
│   ├── model/                # ModelConfig + ModelLoader
│   ├── tokenizer/            # GPT-2 BPE
│   └── server/               # HTTP Server
├── src/                      # 实现文件
│   ├── ops/                  # 12 类 CUDA kernel
│   ├── kv_cache/             # paged_attention.cu, block_allocator.cpp
│   ├── engine/               # engine_server.cpp, scheduler.cpp, batch_loop.cpp
│   └── ...
├── tests/                    # 19 个测试文件
├── tools/                    # 基准测试工具
├── docs/                     # LaTeX 开发手册
├── CMakeLists.txt
└── README.md
```

## GPU 算子

| 算子 | 内核数 | 技术 |
|------|:---:|------|
| GEMM (fp32) | 4 | Naive(430) → Tiled(481) → Vectorized f4(1646) → DoubleBuf(1502) GFLOPS |
| GEMM (fp16) | 1 | cuBLAS Tensor Core (~4.7 TFLOPS) |
| PagedAttention (decode) | 3 | 基线 + Float4/DoubleBuf + Split-K |
| PagedAttention (prefill) | 2 | 直接内核 + tiled FlashAttention 变体 |
| RMSNorm | 1 | warp-reduce + cross-warp |
| RoPE | 1 | Neox-style, GPU cos/sin |
| Softmax | 1 | online-safe warp-reduce |
| Element-wise | 1 | add/mul/scale/scalar |
| Activation | 7 | silu/gelu/relu/sigmoid/tanh/swish/hardswish |
| Unary | 5 | exp/log/sqrt/neg/reciprocal |
| Reduce | 4 | sum/max/min/mean + topk |
| Embedding | 1 | gather |
| lm_head | 1 | GPU GEMV, shared memory |

## 技术亮点

### PagedAttention 优化

- **Float4 向量化**：128-bit 加载替代 32-bit 标量，带宽利用率提升 4x
- **双缓冲**：ping-pong 共享内存 tile，加载与计算重叠
- **Split-K**：长序列 (>256 tokens) 划分到 4 个子网格并行，reduce 合并部分 softmax

### Paged Prefill Attention

直接通过 block table 读取 KV，无需重建连续张量。相比逐 token cudaMemcpy 方案加速 6-47x，消除并发用户数限制。

### Radix Tree 前缀缓存

SGLang 风格的 token 级前缀匹配，支持节点分裂、ref_count 共享、LRU 淘汰。
可通过环境变量配置：`LIGHTLLM_PREFIX_CACHE=hash|radix`。

### Continuous Batching

DecodeFirst 调度器：decode token 优先，prefill 使用剩余 token budget。
长 prompt 通过分块 prefill 穿插执行，不阻塞交互用户。

### 跨平台验证与稳定性

- **零改动跨平台编译**：Linux (GCC 11 + CUDA 11.8) 与 Windows (MSVC + CUDA 13.2) 编译 111/111 目标一致通过；19 测试 / ~202k 断言双平台 0 失败
- **悬垂指针 bug 修复**：Tensor `view()` 自赋值导致 use-after-free，compute-sanitizer 定位到 rope_kernel 越界（1124 → 0 错误）；新增 `reshape_inplace()` 修复，并发上限 32 → 512 (16×)
- 详见 `docs/BUG_ANALYSIS_view-dangling.md`

## License

MIT

## 参考

- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention, Continuous Batching
- [SGLang](https://github.com/sgl-project/sglang) — RadixAttention, 前缀缓存
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — Tiled online softmax
