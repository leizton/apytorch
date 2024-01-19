#pragma once

#include <onnx/onnx_pb.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

std::string prettyPrint(const ::ONNX_NAMESPACE::ModelProto& model);

} // namespace jit
} // namespace torch
