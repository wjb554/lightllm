#pragma once
/// Type-dispatch macros for CUDA kernels using __VA_ARGS__ (comma-safe).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

#include "lightllm/tensor.h"

namespace lightllm {
namespace ops {
inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
inline int next_pow2(int x) { int p = 1; while (p < x) p <<= 1; return p; }
}  // namespace ops
}  // namespace lightllm

// Variadic dispatch — all commas inside __VA_ARGS__ are preserved.
#define DISPATCH_FLOAT_TYPES(DTYPE, NAME, ...)                      \
    [&] {                                                           \
        using namespace lightllm;                                   \
        switch (DTYPE) {                                            \
            case DType::F32: {                                      \
                using scalar_t = float;                             \
                __VA_ARGS__                                         \
                break;                                              \
            }                                                       \
            case DType::F16: {                                      \
                using scalar_t = half;                              \
                __VA_ARGS__                                         \
                break;                                              \
            }                                                       \
            case DType::BF16: {                                     \
                using scalar_t = __nv_bfloat16;                     \
                __VA_ARGS__                                         \
                break;                                              \
            }                                                       \
            default:                                                \
                throw std::runtime_error(                           \
                    std::string(NAME) + ": unsupported dtype");     \
        }                                                           \
    }()

#define DISPATCH_FLOAT_HALF(DTYPE, NAME, ...)                       \
    [&] {                                                           \
        using namespace lightllm;                                   \
        switch (DTYPE) {                                            \
            case DType::F32: {                                      \
                using scalar_t = float;                             \
                __VA_ARGS__                                         \
                break;                                              \
            }                                                       \
            case DType::F16: {                                      \
                using scalar_t = half;                              \
                __VA_ARGS__                                         \
                break;                                              \
            }                                                       \
            default:                                                \
                throw std::runtime_error(                           \
                    std::string(NAME) + ": unsupported dtype");     \
        }                                                           \
    }()
