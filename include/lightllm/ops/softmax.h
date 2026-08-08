#pragma once
#include "lightllm/tensor.h"

namespace lightllm {
namespace ops {

Tensor softmax(const Tensor& x);

}  // namespace ops
}  // namespace lightllm
