#pragma once
/// LightLLM CUDA memory allocator.
///
/// Thin wrapper around cudaMalloc / cudaFree with optional memory-pool
/// support (cudaMemPool API, CUDA 11.2+).  Tracks usage statistics.

#include <cstddef>
#include <cstdint>

namespace lightllm {

class CudaAllocator {
public:
    explicit CudaAllocator(size_t pool_size_bytes = 0);
    ~CudaAllocator();

    // Allocate / free device memory
    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    // Statistics
    size_t used_bytes() const { return used_bytes_; }
    size_t peak_bytes() const { return peak_bytes_; }
    size_t num_allocations() const { return num_allocs_; }

private:
    size_t used_bytes_ = 0;
    size_t peak_bytes_ = 0;
    size_t num_allocs_ = 0;
    bool use_pool_ = false;
};

}  // namespace lightllm
