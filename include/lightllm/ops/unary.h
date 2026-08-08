#pragma once
/// Element-wise unary / math ops — in-place on GPU.

#include "lightllm/tensor.h"

namespace lightllm {
namespace ops {

void neg(Tensor& x);
void abs_op(Tensor& x);       // abs is macro in stdlib
void sign(Tensor& x);

void clip(Tensor& x, float lo, float hi);
void round_op(Tensor& x);
void floor_op(Tensor& x);
void ceil_op(Tensor& x);

void pow_op(Tensor& x, float exponent);
void sqrt_op(Tensor& x);
void exp_op(Tensor& x);
void log_op(Tensor& x);

}  // namespace ops
}  // namespace lightllm
