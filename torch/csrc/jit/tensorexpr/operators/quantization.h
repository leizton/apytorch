#pragma once

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {
namespace tensorexpr {

ExprHandle quantizePerTensorQParamFromArg(ArgValue arg);

double immQScale(const BufHandle& qx);

int64_t immQZero(const BufHandle& qx);

ScalarType immQDType(const BufHandle& qx);

bool isQuantized(const BufHandle& qx);

Tensor computeQuantizePerTensor(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizePerTensorExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedConv1d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedConv2dPrepack(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedConv1d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedConv2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedConv2dRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedLinear(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedLinearRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedAdd(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedAddExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedMul(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedMulScalar(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedCat(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeDequantize(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeDequantizeExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeUpsampleNearest2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeUpsampleNearest2dExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeQuantizedSigmoidExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device);
} // namespace tensorexpr
} // namespace jit
} // namespace torch
