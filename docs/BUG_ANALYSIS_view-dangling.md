# Bug 分析：Tensor view 自赋值导致悬垂指针

> 复现环境：RTX 4090 + CUDA 11.8（云服务器），RTX 2060 + CUDA 13.2（本机）
> 触发场景：并发用户 ≥ 48（`bench_max_users`）
> 影响：跨平台、跨架构一致复现

## 一、问题现象

运行 `bench_max_users`（逐级增加并发用户）时，32 用户正常，48 用户崩溃：

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  cudaMalloc: an illegal memory access
Aborted (core dumped)
```

**关键线索**：`cudaMalloc: illegal memory access` 不是 cudaMalloc 本身失败，而是**某个 GPU kernel 越界访问**污染了 CUDA context，导致后续所有 CUDA 调用（包括 cudaMalloc）报错。真正的越界发生在更早的 kernel。

## 二、定位过程

### 1. compute-sanitizer 定位越界 kernel

```bash
compute-sanitizer --tool memcheck --print-limit 5 ./build/bench_max_users
```

输出：
```
========= Invalid __global__ read of size 4 bytes
=========     at 0x3f0 in void lightllm::ops::rope_kernel<float>(...)
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7f640548c000 is out of bounds
=========     and is 1 bytes after the nearest allocation
=========         at 0x7f6405446000 of size 286720 bytes
```

- 越界 kernel：`rope_kernel`（RoPE 旋转位置编码）
- 越界地址：某块 **286,720 字节** 分配内存的末尾 +1
- 286,720 bytes = 80 tokens × 896 (hidden dim) × 4 = **Q 投影后的 flat 张量**

### 2. 地址步进规律揭示真凶

前 3 个错误的地址：
```
thread(0) → 0x...c000
thread(1) → 0x...c104   (+0x104 = 260 bytes)
thread(2) → 0x...c208   (+0x104 = 260 bytes)
```

每线程地址步进 65 floats（64 = head 步长 + 1 = i 步长），对应 K rope 的 head 索引映射。**但参数检查完全正确**（q: heads=14 stok=896；k: heads=2 stok=128）。

### 3. 关键发现：指针悬垂

在 kernel 开头加调试打印：
```
ROPE: tokens=80 heads=14 hd=64 stok=896 shd=64 ptr=0x7ff1c948c000
```

**`ptr=0x...c000` 恰好等于 compute-sanitizer 报的越界地址**（0x...6000 分配内存的末尾）。

**q_flat.data() 指向了自己分配内存的末尾** —— 这是悬垂指针。

## 三、根因

### Tensor 的 view 语义缺陷

`include/lightllm/tensor.h`：
```cpp
// The returned Tensor does NOT own the data.
Tensor view(std::vector<int> new_shape) const;
```

`view()` 返回一个**不拥有数据**的临时 Tensor（`owns_data_ = false`），与源共享同一块 buffer。

### Engine 中的危险用法

`src/engine/engine_server.cpp`：
```cpp
Tensor q_flat = gemm(normed, L.q, true);   // [T, Hq*hd]，owns_data_ = true，buffer = B
q_flat = q_flat.view({total_tokens, Hq_, hd_});  // ← BUG！
```

执行 `q_flat = q_flat.view(...)` 时：

```
1. q_flat.view(...) 创建临时 view Tensor V
   V.data_ = B（同一个 buffer）
   V.owns_data_ = false（view 不拥有）

2. q_flat = V 触发 move 赋值
   move 赋值释放 q_flat 旧的 buffer B   ← 关键！
   然后接管 V 的 data_（还是 B，但已释放！）

3. V 析构（owns_data_=false，不释放）
   q_flat.data() = B = 已释放的悬垂指针
```

**自赋值 + view 共享 buffer = 释放了仍在使用的内存。** 后续任何 kernel（如 rope）读 q_flat 就访问已释放/重分配的内存 → 越界。

### 为什么 32 OK 48 崩

这是**内存布局的偶然性**——小 batch 时释放的内存未被立即重用，rope 读到的还是旧值（碰巧没崩）。大 batch（≥48）时 GPU 内存分配器重用该块，rope 读到错误数据 → 越界。

**本质是未定义行为（use-after-free），任何规模都可能崩，只是阈值处必现。**

## 四、修复

### 修复方案：in-place reshape

给 Tensor 增加 `reshape_inplace()`，只改 shape 不碰 buffer：

`include/lightllm/tensor.h`：
```cpp
// Reshape IN-PLACE (must preserve numel). Safe for self-assignment
// patterns like `t = t.view(...)` which would otherwise double-free.
void reshape_inplace(std::vector<int> new_shape);
```

`src/tensor.cpp`：
```cpp
void Tensor::reshape_inplace(std::vector<int> new_shape) {
    size_t new_numel = 1;
    for (int d : new_shape) new_numel *= d;
    if (new_numel != numel()) {
        throw std::runtime_error(
            "reshape_inplace: shape has different numel");
    }
    shape_ = std::move(new_shape);
}
```

### 修改 7 处自赋值调用点

```cpp
// 修复前（悬垂）
q_flat = q_flat.view({total_tokens, Hq_, hd_});
k_flat = k_flat.view({total_tokens, Hkv_, hd_});
v_flat = v_flat.view({total_tokens, Hkv_, hd_});
attn_out = attn_out.view({P, D_});
attn_out = attn_out.view({1, D_});

// 修复后（安全）
q_flat.reshape_inplace({total_tokens, Hq_, hd_});
k_flat.reshape_inplace({total_tokens, Hkv_, hd_});
v_flat.reshape_inplace({total_tokens, Hkv_, hd_});
attn_out.reshape_inplace({P, D_});
attn_out.reshape_inplace({1, D_});
```

修改文件：
- `include/lightllm/tensor.h` — 新增声明
- `src/tensor.cpp` — 新增实现
- `src/engine/engine_server.cpp` — 3 处（q/k/v）
- `src/engine/engine.cpp` — 2 处（attn_out）
- `src/engine/engine_paged.cpp` — 2 处（attn_out）

## 五、修复验证

### compute-sanitizer：1124 错误 → 0 错误

```
修复前: ERROR SUMMARY: 1124 errors
修复后: ERROR SUMMARY: 0 errors
```

### 并发上限：32 → 512

```
修复前: 32 OK, 48 崩溃
修复后: 512 OK（16→512 全部通过，blocks 全程 2056 free 无泄漏）
```

### 功能无回归

| 测试 | 修复前 | 修复后 |
|------|:---:|:---:|
| test_generate | 15-17 tok/s | 16.9 tok/s |
| test_concurrent | 25.1 tok/s | 25.1 tok/s |
| test_mixed_users | 3 passed | 3 passed |

## 六、经验总结

1. **GPU 崩溃的错误信息会骗人**：`cudaMalloc: illegal memory access` 通常不是 cudaMalloc 的错，而是之前某 kernel 越界污染了 context。要用 compute-sanitizer 穿透定位真正的越界 kernel。

2. **地址步进规律 = 诊断利器**：对比相邻 thread 的越界地址差值，能反推访问模式（stride 步长），缩小到具体 kernel 内的索引计算。

3. **自赋值是 C++ 的经典陷阱**：`t = t.view(...)` 这种自引用赋值，配合"不拥有数据的 view"语义，必然造成 use-after-free。规则：**有所有权语义的对象，自赋值必须先检测 `data_` 是否相同，或用 in-place 操作。**

4. **修复后的数据**：并发上限从 32 提升到 512（16×），这是本项目最大的隐藏 bug 之一——影响所有多用户并发推理场景。
