# LightLLM 服务器性能报告 (RTX 4090)

> 测试日期：2026-08-05
> 测试环境：云服务器 · RTX 4090 24GB · CUDA 11.8 · 16核 Xeon Gold 6430 · Ubuntu
> 模型：Qwen2.5-0.5B (fp32)
> 对比基准：本机 RTX 2060 (CUDA 13.2)

## 测试环境

| 项目 | 本机 (基准) | 服务器 |
|------|:---:|:---:|
| GPU | RTX 2060 6GB | RTX 4090 24GB |
| CUDA | 13.2 | 11.8 |
| 编译器 | MSVC 2026 | GCC 11 |
| 系统 | Windows 11 | Ubuntu 22.04 |
| GPU 架构 | sm_75 | sm_89 |

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

## 生成性能对比

### 单用户 / 并发

| 场景 | RTX 2060 | RTX 4090 | 提升 |
|------|:---:|:---:|:---:|
| 单用户生成 | 18.7 tok/s | 22.9 tok/s | 1.22× |
| 3 用户并发 | 25.3 tok/s | **103.7 tok/s** | **4.1×** |
| 16 用户混合长度 | 19.8 tok/s | ~40 tok/s | 2× |

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
| 512 | 待确认 | **OK** |

**RTX 4090 稳定支持 512 并发用户**（修复前 32 即崩溃）。

## 结论

1. **跨平台验证**：代码在 Linux + CUDA 11.8 + GCC 下编译 111/111 通过，无需修改（C++17 自动降级）。
2. **并发能力**：3 用户批量达 103.7 tok/s（4.1× 于 2060），修复悬垂指针后 512 并发稳定。
3. **延迟改善**：TPOT 385ms vs 1558ms（4×），TTFT 879ms vs 3149ms（3.6×），交互体验显著提升。
4. **吞吐上限**：~30 tok/s 受限于 max_batch_tokens=512 与 fp32 模型大小，GPU 算力未吃满。

## 已知限制

- fp32 模式未利用 Tensor Core（CUDA 11.8 有 WMMA，可在 4090 实现手写 fp16 GEMM）。
- 0.5B 模型计算量小，无法体现 4090 全部算力。
- 悬垂指针 bug 已修复（详见 `BUG_ANALYSIS_view-dangling.md`）。

## 附录

- 本机对照：`REPORT_rtx2060.md`
- Bug 分析：`BUG_ANALYSIS_view-dangling.md`
- 开发手册：`lightllm-dev-manual.tex`
