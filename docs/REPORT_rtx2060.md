# LightLLM 本地性能报告 (RTX 2060)

> 测试日期：2026-08-05
> 测试环境：Windows 11 · RTX 2060 6GB · CUDA 13.2 · MSVC 2026
> 模型：Qwen2.5-0.5B (fp32)
> 对照：`REPORT_rtx4090.md` (云端 RTX 4090)

## 测试环境

| 项目 | 本机 (2060) | 服务器 (4090) |
|------|:---:|:---:|
| GPU | RTX 2060 6GB | RTX 4090 24GB |
| CUDA | 13.2 | 11.8 |
| 编译器 | MSVC 2026 | GCC 11 |
| 系统 | Windows 11 | Ubuntu 22.04 |
| GPU 架构 | sm_75 (Turing) | sm_89 (Ada) |

## 正确性测试

19 个测试全部通过，合计 ~202,000 断言 0 失败：

| 测试 | 断言 | 状态 |
|------|------|:---:|
| test_tensor | 19 | PASS |
| test_ops | 57,868 | PASS |
| test_quality | 2,098 | PASS |
| test_sampling | 140,612 | PASS |
| test_lm_head | 135 | PASS |
| test_paged_attention | 523 | PASS |
| test_paged_attention_v2 | 166 | PASS |
| test_paged_prefill | 147 | PASS |
| test_prefix_cache | 176 | PASS |
| test_radix_engine | 244 | PASS |
| test_batch_loop | 139 | PASS |
| test_server_engine | 108 | PASS |
| test_mixed_users | 3 | PASS |

## 算子性能

| 算子 | 性能 |
|------|------|
| GEMM fp32 手写 (DoubleBuf) | 855 GFLOPS |
| GEMM fp32 cuBLAS | 1791 GFLOPS |
| GEMM fp16 cuBLAS (Tensor Core) | ~3.2 TFLOPS (720us) |
| lm_head kernel (D=896, V=151936) | 17.2 ms, 31.7 GB/s |

## 生成性能

### 单用户 / 并发

| 场景 | RTX 2060 | RTX 4090 | 提升 |
|------|:---:|:---:|:---:|
| 单用户生成 | 15.8 tok/s | 22.9 tok/s | 1.45× |
| 3 用户并发 | 25.1 tok/s | **103.7 tok/s** | **4.1×** |

### 高吞吐仿真（60s, 3 req/s, 180 请求）

| 指标 | RTX 2060 | RTX 4090 | 提升 |
|------|:---:|:---:|:---:|
| 稳态吞吐 | 24.0 tok/s | 30.3 tok/s | 1.26× |
| P50 TTFT | 3149 ms | 879 ms | **3.6×** |
| P50 TPOT | 1558 ms | 385 ms | **4.0×** |
| P99 TTFT | 4565 ms | 1067 ms | 4.3× |
| 请求完成 | 180 (0 失败) | 180 (0 失败) | — |

### 并发扩展性（悬垂指针修复后）

| 用户数 | RTX 2060 | RTX 4090 |
|:---:|:---:|:---:|
| 16 | OK | OK |
| 32 | OK | OK |
| 48 | OK | OK |
| 64 | OK | OK |
| 96 | OK | OK |
| 128 | OK | OK |
| 192 | OK | OK |
| 256 | OK | OK |
| 384 | OK | OK |
| 512 | 待确认 | OK |

**RTX 2060 修复后稳定支持 384+ 并发用户**（修复前 32 即崩溃）。

## 结论

1. **正确性**：RTX 2060 与 4090 上 19 测试全过，跨平台、跨架构一致。
2. **并发收益**：3 用户批量 25.1 tok/s vs 单用户 15.8 tok/s（1.6×），证明 Continuous Batching 有效。
3. **4090 优势**：3 用户并发 4.1×、TPOT 4.0× 快，批量场景算力差距显著。
4. **本机限制**：CUDA 13.2 移除了 sm_75 的 WMMA，无法手写 Tensor Core GEMM（4090 + CUDA 11.8 可实现）。
5. **Bug 修复**：`t = t.view()` 悬垂指针 bug 已修复，并发上限从 32 提升到 384+（详见 BUG 分析）。

## 已知限制

- fp32 模式未利用 Tensor Core。
- 0.5B 模型计算量小，无法体现两卡全部算力。

## 附录

- 服务器对照：`REPORT_rtx4090.md`
- Bug 分析：`BUG_ANALYSIS_view-dangling.md`
- 开发手册：`lightllm-dev-manual.tex`
