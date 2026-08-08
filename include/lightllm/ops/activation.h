#pragma once
/// Activation functions — all in-place on GPU.
/// Supports fp32 and fp16 via DISPATCH_FLOAT_TYPES.

#include "lightllm/tensor.h"

namespace lightllm {
namespace ops {

void relu(Tensor& x);
void sigmoid(Tensor& x);
void tanh_op(Tensor& x);     // tanh taken by <cmath>
void gelu(Tensor& x);
void swish(Tensor& x);       // = SiLU, alias
void mish(Tensor& x);

}  // namespace ops
}  // namespace lightllm
