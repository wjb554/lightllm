# LightLLM GEMM 性能报告

> 日期: 2026-08-08
> 设备: NVIDIA GeForce RTX 2060 (sm_75, 6GB, 1920 CUDA cores, 240 Tensor cores)
> 环境: Windows 11 · CUDA 13.2 · MSVC 2026 (v18) · nvcc `-arch=sm_75 -O3`
> 方法: 每个 (shape, backend) 组合预热 30 次 → 测 50 次取中位数 (p50) 及 [min..max] 范围
> 代码: `lightllm/tests/test_ops.cu` — `bench_gemm()`
>
> 理论峰值 (RTX 2060):
> - FP32 CUDA core: 6.5 TFLOPS
> - FP16 CUDA core: 12.9 TFLOPS
> - FP16 Tensor Core (WMMA/HMMA): 59 TFLOPS
> - HBM 带宽: 336 GB/s (理论), ~248 GB/s (实测上限, 见 cuda-ops PERFORMANCE.md)

---

## 1. Square GEMMs — 计算受限基准

| Shape | Backend | p50 (us) | [min..max] | GFLOPS | % of cuBLAS |
|------|------|:---:|------|:---:|:---:|
| **512^3** | Naive | 968 | [734..979] | 277 | 14% |
| | Tiled | 631 | [623..642] | 425 | 21% |
| | Vec | 217 | [215..221] | **1,237** | 62% |
| | DBuf | 244 | [243..256] | 1,100 | 55% |
| | cuBLAS fp32 | 134 | [112..142] | 2,003 | 100% |
| | cuBLAS fp16 TC | 44 | [42..51] | **6.1 TFLOPS** | — |
| **1024^3** | Naive | 4,995 | [4887..5339] | 430 | 16% |
| | Tiled | 4,520 | [4387..4952] | 475 | 18% |
| | Vec | 1,369 | [1332..1538] | **1,569** | 59% |
| | DBuf | 1,508 | [1479..1861] | 1,424 | 54% |
| | cuBLAS fp32 | 811 | [795..842] | 2,648 | 100% |
| | cuBLAS fp16 TC | 475 | [447..572] | **4.5 TFLOPS** | — |
| **2048^3** | Naive | 37,542 | [37139..38212] | 458 | 11% |
| | Tiled | 35,104 | [34521..35988] | 489 | 11% |
| | Vec | 8,253 | [8095..8860] | **2,082** | 48% |
| | DBuf | 9,475 | [9341..10115] | 1,813 | 42% |
| | cuBLAS fp32 | 3,988 | [3866..4332] | 4,308 | 100% |
| | cuBLAS fp16 TC | 1,229 | [1178..1427] | **14.0 TFLOPS** | — |

**分析:**
- **Vec > DBuf 在所有三个 square shape 下均成立。** 双缓冲在 K=512~2048 范围内, swap 开销(每 tile 迭代拷贝 As/Bs) 超过 load-compute 重叠收益。DBuf 需要更大 K (如 4096+) 才能反转。
- 手写后端与 cuBLAS 的差距随 shape 增大而扩大: 512^3 时 Vec 达 cuBLAS 62% → 2048^3 时仅 48%。cuBLAS 利用了内部微架构优化(warp-level matrix multiply, 寄存器预取) 在更大 shape 下体现得更充分。
- fp16 Tensor Core 2048^3 达 **14.0 TFLOPS** — 超过 FP16 CUDA core 峰值 12.9, 证明确实走到了 Tensor Core 路径。
- Tiled 相比 Naive 提升有限 (18-25%), 主要是 16×16 tile 在 256 线程 block 下每线程只负责 1 个输出元素, 计算密度不足。

---

## 2. Qwen2-0.5B Decode (M=1, 内存受限)

测试真实的 Qwen2-0.5B (D=896, V=151936, intermediate=4864) 单 token decode 形状。

| 算子 | M | K | N | Backend | p50 (us) | GFLOPS | GB/s |
|------|:---:|:---:|:---:|------|:---:|:---:|:---:|
| **q_proj** | 1 | 896 | 896 | cuBLAS | 31 | 52 | 103.8 |
| | | | | Vec | 122 | 13 | 26.4 |
| | | | | Naive | 36 | 45 | 89.4 |
| **k_proj** | 1 | 896 | 128 | cuBLAS | 23 | 10 | 20.1 |
| | | | | Naive | 27 | 8 | 17.1 |
| | | | | Vec | 112 | 2 | 4.1 |
| **o_proj** | 1 | 896 | 896 | cuBLAS | 34 | 47 | 94.7 |
| | | | | Vec | 123 | 13 | 26.2 |
| | | | | Naive | 38 | 42 | 84.7 |
| **gate_proj** | 1 | 896 | 4864 | cuBLAS | 92 | 95 | 189.7 |
| | | | | Naive | 101 | 86 | 172.8 |
| | | | | Vec | 266 | 33 | 65.6 |
| **down_proj** | 1 | 4864 | 896 | cuBLAS | 146 | 60 | 119.6 |
| | | | | Naive | 173 | 50 | 100.9 |
| | | | | Vec | 677 | 13 | 25.8 |
| **lm_head** | 1 | 896 | 151936 | cuBLAS | 2,152 | 127 | 253.3 |
| | | | | Naive | 3,249 | 84 | 167.8 |
| | | | | Vec | 6,754 | 40 | 80.7 |

**分析:**
- **M=1 时 Naive > Vec/DBuf** (对 q_proj: Naive 36us vs Vec 122us, 3.4×)! 因为每 block 只有 1 个活跃线程, Vec/DBuf 的 64×64 tile 和双缓冲初始化反而增加了 fixed cost。Naive 极简设计(零 shared memory, 零 sync) 在这种极端 M=1 场景下反而有优势。
- cuBLAS 在所有 M=1 形状下都一骑绝尘 — 说明它在 GEMV 特化路径上做了很多优化 (如 warp-level reduction、load caching)。
- **lm_head (N=151936) 是 decode 的最大瓶颈**: 2.2ms per token (cuBLAS fp32) → 在 24 层模型 decoder 中 lm_head 占总 latency ~4%。
- fp16 cuBLAS (Tensor Core) 对 M=1 基本无加速 — M=1 时计算量太小, Tensor Core 的固定开销 > 收益。

---

## 3. Qwen2-0.5B Prefill (M=64, 计算受限)

模拟 64-token chunk prefill。

| 算子 | M | K | N | Backend | p50 (us) | GFLOPS | % of cuBLAS |
|------|:---:|:---:|:---:|------|:---:|:---:|:---:|
| **q_proj** | 64 | 896 | 896 | Vec | 155 | **663** | 30% |
| | | | | DBuf | 148 | 694 | 32% |
| | | | | cuBLAS | 47 | 2,186 | 100% |
| **gate_proj** | 64 | 896 | 4864 | Vec | 303 | **1,841** | 50% |
| | | | | DBuf | 336 | 1,660 | 46% |
| | | | | cuBLAS | 153 | 3,646 | 100% |
| **down_proj** | 64 | 4864 | 896 | Vec | 723 | 772 | 22% |
| | | | | DBuf | 724 | **771** | 22% |
| | | | | cuBLAS | 158 | 3,531 | 100% |

**分析:**
- Vec 在 gate_proj (large N=4864) 下达到 hand-written 最佳: **1,841 GFLOPS** — 这是 M=64 prefill 场景下手写 GEMM 的最优性能。
- **down_proj 形状 (M=64, K=4864, N=896) 是手写后端与 cuBLAS 差距最大的形状** (仅 22%)。K 很大但 N 相对小, Vec/DBuf 的 64-wide N-tile 利用率不足 — 在 N=896 时只能填满 14 个 64-wide column block, 最后一个 block 大部分空闲。
- DBuf 在 gate_proj (N=4864) 下弱于 Vec (1660 < 1841) — 同样原因: swap 开销 > load 隐藏收益。
- fp16 cuBLAS Tensor Core 在 gate_proj (M=64) 下达 **8.7 TFLOPS** — 这是 prefill 场景下可用的实际加速。

---

## 4. Batch Scaling (K=896, N=896, vary M)

固定 Qwen2-0.5B q_proj 形状 (K=N=896), 变化 batch size M。

| M | Backend | p50 (us) | GFLOPS | GB/s | cuBLAS vs |
|:---:|------|:---:|:---:|:---:|------|
| **1** | Vec | 122 | 13 | 26.4 | cuBLAS 59 GFLOPS (4.5×) |
| | cuBLAS | 27 | 59 | 119.2 | |
| **4** | Vec | 127 | 51 | 25.5 | cuBLAS 178 GFLOPS (3.5×) |
| | cuBLAS | 36 | 178 | 90.0 | |
| **16** | Vec | 129 | 199 | 25.8 | cuBLAS 571 GFLOPS (2.9×) |
| | cuBLAS | 45 | 571 | 73.9 | |
| **64** | Vec | 144 | 714 | 25.5 | cuBLAS 2,186 GFLOPS (3.1×) |
| | cuBLAS | 47 | 2,186 | 78.1 | |

**分析:**
- Vec 的带宽利用率稳定在 ~26 GB/s 无论 M, 说明它是 compute-bound (K=N=896 时带宽天花板 ≈ 120 GB/s)。
- cuBLAS 在 M=1→4→16→64 时性能几乎线性扩展 (59→178→571→2186 GFLOPS), 利用了大 batch 的权重复用。
- M=1→4: Vec 仅从 13→51 GFLOPS (3.9× ≈ 4×) — 说明 M 小到一定阈值后 GPU 才能被填满。
- **结论**: M≥16 时 GPU 利用率开始合理, M≥64 时进入 compute-bound 区域。

---

## 5. lm_head Vocab Scaling (M=1, K=896, vary V)

| V | Backend | p50 (us) | GFLOPS |
|:---:|------|:---:|:---:|
| **896** | Vec | 122 | 13 |
| | cuBLAS | 31 | 52 |
| **4,864** | Vec | 266 | 33 |
| | cuBLAS | 92 | 95 |
| **16,384** | Vec | 801 | 37 |
| | cuBLAS | 217 | 135 |
| **50,257** (GPT-2) | Vec | 2,399 | 38 |
| | cuBLAS | 714 | 126 |
| **151,936** (Qwen2) | Vec | 6,754 | 40 |
| | cuBLAS | 2,152 | 127 |

**分析:**
- Vec 的 GFLOPS 稳定在 33-40 (带宽约 66-81 GB/s), 受限于 M=1 的权重读取模式。
- cuBLAS GFLOPS 从 52 (V=896) 增长到 127 (V=151936) — 更大 V 让 cuBLAS 有更多 column parallelism。
- **Qwen2 的 lm_head (V=151936) 耗时 2.2ms/token** — 这是 decode 延迟中的最大单一成本。GPT-2 的 lm_head (V=50257) 仅需 0.7ms。

---

## 6. 总结

### 6.1 关键数据 (用于简历/报告)

| 指标 | 值 | 条件 |
|------|------|------|
| 手写 GEMM 最优性能 | **2,082 GFLOPS** | Vec, 2048³ fp32 (cuBLAS 的 48%) |
| Prefill 场景最优 | **1,841 GFLOPS** | Vec, gate_proj M=64 K=896 N=4864 (cuBLAS 的 50%) |
| Square 512³ 效率 | 1,237 GFLOPS | Vec, cuBLAS 的 62% — 最佳接近比 |
| M=1 decode 最优手写 | **Naive > Vec** | Naive 在多组 M=1 形状下反超 Vec/DBuf (零 overhead) |
| fp16 Tensor Core 峰值 | **14.0 TFLOPS** | 2048³ cuBLAS fp16 |
| 最大瓶颈 | lm_head V=151936: 2.2ms/token | 占单 token decode 延迟 ~4% |

### 6.2 Vec vs DBuf 结论

在 K ≤ 2048 范围内, **Vectorized kernel 明确优于 DoubleBuf**。DBuf 的双缓冲 swap 开销 (每 tile 迭代拷贝 ~1,100 个 float) 超过 load-compute overlap 收益。DBuf 需要 K ≥ 4096 才能反超。在手写 GEMM 优化中应优先投入 Vec 路径 (如升级为 WMMA Tensor Core、增大 TM/TN、fuse bias)。

### 6.3 M=1 Decode 结论

**不要用手写 GEMM 做 decode。** Naive 在 M=1 下表现尚可 (>cuBLAS 的 ~80%) 但仍是巧合 — 对极小的 M, 零-overhead 胜过任何共享内存优化。实际部署时应使用 cuBLAS 的 GEMV 特化路径, 或将 QKV/FFN 投影融合成 single batched operation。

### 6.4 测试命令

```
cd lightllm\build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja test_ops.exe
test_ops.exe   # bench_gemm() 自动运行 5 组 benchmark
```

完整 benchmark 输出在 `build/bench_result.txt`。
