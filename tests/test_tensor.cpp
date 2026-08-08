// LightLLM - Tensor unit tests
#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>

#include "lightllm/tensor.h"

using lightllm::Device;
using lightllm::DType;
using lightllm::Tensor;

// helpers
static int tests_run = 0;
static int tests_passed = 0;

static void check(const char* name, bool cond) {
    tests_run++;
    if (cond) tests_passed++;
    else std::fprintf(stderr, "  FAIL: %s\n", name);
}

template <typename T>
static bool allclose(const std::vector<T>& a, const std::vector<T>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (std::fabs(static_cast<float>(a[i]) - static_cast<float>(b[i])) > tol)
            return false;
    }
    return true;
}

static void test_cpu_tensor() {
    Tensor t({2, 3}, DType::F32, Device::CPU);
    check("cpu_shape", t.shape().size() == 2 && t.shape()[0] == 2 && t.shape()[1] == 3);
    check("cpu_numel", t.numel() == 6);
    check("cpu_bytes", t.nbytes() == 6 * 4);
    check("cpu_device", t.is_cpu());
    check("cpu_defined", t.defined());

    std::vector<float> src = {1, 2, 3, 4, 5, 6};
    t.copy_from(src.data(), t.nbytes());

    std::vector<float> dst(6);
    t.copy_to(dst.data(), t.nbytes());
    check("cpu_roundtrip", allclose(src, dst, 1e-6f));

    std::printf("  [PASS] test_cpu_tensor\n");
}

static void test_gpu_tensor() {
    Tensor t({4, 128}, DType::F32, Device::CUDA);
    check("gpu_shape", t.shape()[0] == 4 && t.shape()[1] == 128);
    check("gpu_numel", t.numel() == 512);
    check("gpu_bytes", t.nbytes() == 2048);
    check("gpu_device", t.is_cuda());
    check("gpu_defined", t.defined());

    std::vector<float> src(512);
    for (size_t i = 0; i < 512; i++) src[i] = static_cast<float>(i) * 0.1f;
    t.copy_from(src.data(), t.nbytes());

    std::vector<float> dst(512);
    t.copy_to(dst.data(), t.nbytes());
    check("gpu_roundtrip", allclose(src, dst, 1e-3f));

    std::printf("  [PASS] test_gpu_tensor\n");
}

static void test_move_semantics() {
    Tensor a({3, 4}, DType::F32, Device::CPU);
    void* original_ptr = a.data<float>();

    Tensor b = std::move(a);
    check("move_defined", b.defined());
    check("move_not_defined", !a.defined());
    check("move_same_ptr", b.data<float>() == original_ptr);

    std::printf("  [PASS] test_move_semantics\n");
}

static void test_view() {
    Tensor a({2, 6}, DType::F32, Device::CPU);
    auto b = a.view({3, 4});

    check("view_shape", b.shape()[0] == 3 && b.shape()[1] == 4);
    check("view_numel", b.numel() == 12);
    check("view_same_ptr", b.data<float>() == a.data<float>());

    std::printf("  [PASS] test_view\n");
}

static void test_fill_zeros() {
    Tensor t({10, 10}, DType::F32, Device::CPU);
    std::vector<float> init(100, 3.14f);
    t.copy_from(init.data(), t.nbytes());
    t.fill_zeros();
    std::vector<float> result(100);
    t.copy_to(result.data(), t.nbytes());
    for (int i = 0; i < 100; i++) {
        if (result[i] != 0.0f) {
            std::fprintf(stderr, "  FAIL: test_fill_zeros at idx %d: %f\n", i, result[i]);
            return;
        }
    }
    check("fill_zeros", true);
    std::printf("  [PASS] test_fill_zeros\n");
}

int main() {
    std::printf("LightLLM - Tensor Tests\n");
    std::printf("=======================\n\n");

    test_cpu_tensor();
    test_gpu_tensor();
    test_move_semantics();
    test_view();
    test_fill_zeros();

    std::printf("\n%d / %d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
