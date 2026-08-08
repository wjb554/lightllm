#include "lightllm/allocator.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace lightllm {

CudaAllocator::CudaAllocator(size_t /*pool_size_bytes*/) {
    // TODO Phase 1b: set up cudaMemPool when pool_size > 0
    use_pool_ = false;
}

CudaAllocator::~CudaAllocator() = default;

void* CudaAllocator::allocate(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    used_bytes_ += bytes;
    if (used_bytes_ > peak_bytes_) peak_bytes_ = used_bytes_;
    num_allocs_++;
    return ptr;
}

void CudaAllocator::deallocate(void* ptr) {
    // We don't track individual allocation sizes here (could be added).
    // cudaFree does not report the freed size, so used_bytes_ is an approximation.
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaFree failed: ") + cudaGetErrorString(err));
    }
}

}  // namespace lightllm
