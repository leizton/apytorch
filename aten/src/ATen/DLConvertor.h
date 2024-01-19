#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/dlpack.h>

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor
// 2) take a dlpack tensor and convert it to the ATen Tensor

namespace at {

ScalarType toScalarType(const DLDataType& dtype);
DLManagedTensor* toDLPack(const Tensor& src);
Tensor fromDLPack(const DLManagedTensor* src);
Tensor
fromDLPack(const DLManagedTensor* src, std::function<void(void*)> deleter);
DLDataType getDLDataType(const Tensor& t);
DLDevice getDLContext(const Tensor& tensor, const int64_t& device_id);

} // namespace at
