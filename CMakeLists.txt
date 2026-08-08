cmake_minimum_required(VERSION 3.20)
project(LightLLM VERSION 0.1.0 LANGUAGES CXX CUDA)

# Default to Release if not specified (e.g., on first configure).
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- CUDA standard: auto-detect based on nvcc version ----
# CUDA 12.x+ fully supports C++20; CUDA 11.x only partially.
# Fall back to C++17 for 11.x to avoid compilation errors.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(CMAKE_CUDA_COMPILER_VERSION AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
    message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION} detected — using C++17 for CUDA (11.x has incomplete C++20 support)")
    set(CMAKE_CUDA_STANDARD 17)
endif()

# ---- GPU architecture ----
# Auto-detect from the GPU if possible, else default to sm_75 (RTX 2060).
# Override with: -DCMAKE_CUDA_ARCHITECTURES="89" (RTX 4090) or "80" (A100)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75")
endif()

# ---- Platform-specific flags ----
if(MSVC)
    # MSVC: force UTF-8 (CXX only, not passed to nvcc)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
else()
    # GCC/Clang: enable warnings, C++20
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
endif()

# Find packages
find_package(CUDAToolkit REQUIRED)

# --- Library: lightllm ---
set(LIGHTLLM_SOURCES
    src/tensor.cpp
    src/allocator.cpp
    src/ops/elementwise.cu
    src/ops/norm.cu
    src/ops/rope.cu
    src/ops/gemm.cu
    src/ops/attention.cu
    src/ops/activation.cu
    src/ops/unary.cu
    src/ops/softmax.cu
    src/ops/reduce.cu
    src/ops/embedding.cu
    src/ops/sampling.cpp
    src/ops/lm_head.cu
    src/model/model_loader.cpp
    # src/model/transformer_block.cpp  # TODO: fix namespace + copy issues
    src/kv_cache/block_allocator.cpp
    src/kv_cache/hash_prefix_cache.cpp
    src/kv_cache/radix_prefix_cache.cpp
    src/kv_cache/paged_attention.cu
    src/server/http_server.cpp
    src/engine/scheduler.cpp
    src/engine/engine_paged.cpp
    src/engine/engine_server.cpp
    src/engine/batch_loop.cpp
    src/tokenizer/tokenizer.cpp
    src/model/model_config.cpp
)

set(LIGHTLLM_HEADERS
    include/lightllm/tensor.h
    include/lightllm/allocator.h
    include/lightllm/ops/dispatch.h
    include/lightllm/ops/elementwise.h
    include/lightllm/ops/norm.h
    include/lightllm/ops/rope.h
    include/lightllm/ops/gemm.h
    include/lightllm/ops/attention.h
    include/lightllm/ops/activation.h
    include/lightllm/ops/unary.h
    include/lightllm/ops/softmax.h
    include/lightllm/ops/reduce.h
    include/lightllm/model/model_loader.h
    include/lightllm/model/model_config.h
    include/lightllm/model/transformer_block.h
    include/lightllm/ops/embedding.h
    include/lightllm/ops/sampling.h
    include/lightllm/ops/lm_head.h
    include/lightllm/kv_cache/prefix_cache.h
    include/lightllm/kv_cache/block_allocator.h
    include/lightllm/kv_cache/paged_attention.h
    include/lightllm/server/http_server.h
    include/lightllm/engine/scheduler.h
    include/lightllm/tokenizer/tokenizer.h
    include/lightllm/engine/engine.h
    include/lightllm/engine/batch_loop.h
)

# engine_paged.cpp + engine_server.cpp contain inline CUDA kernels — compile with nvcc
set_source_files_properties(src/engine/engine_paged.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/engine/engine_server.cpp PROPERTIES LANGUAGE CUDA)

add_library(lightllm STATIC ${LIGHTLLM_SOURCES} ${LIGHTLLM_HEADERS})
target_include_directories(lightllm PUBLIC include)
target_link_libraries(lightllm PUBLIC CUDA::cudart CUDA::cublas)

# Windows: link winsock for HTTP server
if(WIN32)
    target_link_libraries(lightllm PUBLIC ws2_32)
endif()

# --- Tests ---
function(add_lightllm_test test_name test_source)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE lightllm CUDA::cudart CUDA::cublas)
    target_include_directories(${test_name} PRIVATE include)
    set_target_properties(${test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_lightllm_test(test_tensor tests/test_tensor.cpp)
add_lightllm_test(test_ops tests/test_ops.cu)
add_lightllm_test(test_quality tests/test_quality.cu)
add_lightllm_test(test_model tests/test_model.cpp)
add_lightllm_test(test_e2e tests/test_e2e.cpp)
add_lightllm_test(test_full_model tests/test_full_model.cpp)
add_lightllm_test(test_paged_attention tests/test_paged_attention.cu)
add_lightllm_test(test_paged_attention_v2 tests/test_paged_attention_v2.cu)
add_lightllm_test(test_paged_prefill tests/test_paged_prefill.cu)
add_lightllm_test(test_paged_model tests/test_paged_model.cpp)
add_lightllm_test(test_generate tests/test_generate.cu)
add_lightllm_test(test_sampling tests/test_sampling.cpp)
add_lightllm_test(test_lm_head tests/test_lm_head.cu)

# --- Install ---
install(TARGETS lightllm ARCHIVE DESTINATION lib)
install(DIRECTORY include/lightllm DESTINATION include)

add_lightllm_test(test_concurrent tests/test_concurrent.cu)
add_lightllm_test(test_scheduler tests/test_scheduler.cu)
add_lightllm_test(test_tokenizer tests/test_tokenizer.cpp)
add_lightllm_test(test_engine_paged tests/test_engine_paged.cpp)
add_lightllm_test(test_server_engine tests/test_server_engine.cpp)
add_lightllm_test(test_batch_loop tests/test_batch_loop.cpp)
add_lightllm_test(test_prefix_cache tests/test_prefix_cache.cpp)
add_lightllm_test(test_radix_engine tests/test_radix_engine.cpp)
add_lightllm_test(test_mixed_users tests/test_mixed_users.cpp)

# --- Tools ---
add_executable(bench_server tools/bench_server.cpp)
target_link_libraries(bench_server PRIVATE lightllm CUDA::cudart CUDA::cublas)
target_include_directories(bench_server PRIVATE include)

add_executable(bench_simulation tools/bench_simulation.cpp)
target_link_libraries(bench_simulation PRIVATE lightllm CUDA::cudart CUDA::cublas)
target_include_directories(bench_simulation PRIVATE include)
set_target_properties(bench_simulation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bench_throughput tools/bench_throughput.cpp)
target_link_libraries(bench_throughput PRIVATE lightllm CUDA::cudart CUDA::cublas)
target_include_directories(bench_throughput PRIVATE include)
set_target_properties(bench_throughput PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bench_fp16 tools/bench_fp16.cpp)
target_link_libraries(bench_fp16 PRIVATE lightllm CUDA::cudart CUDA::cublas)
target_include_directories(bench_fp16 PRIVATE include)

add_executable(bench_concurrency tools/bench_concurrency.cpp)
target_link_libraries(bench_concurrency PRIVATE lightllm CUDA::cudart CUDA::cublas)
target_include_directories(bench_concurrency PRIVATE include)

add_lightllm_test(bench_mixed_lengths tools/bench_mixed_lengths.cpp)
add_lightllm_test(bench_max_users tools/bench_max_users.cpp)

