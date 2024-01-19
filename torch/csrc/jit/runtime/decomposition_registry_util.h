#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

const std::string& GetSerializedDecompositions();

const OperatorMap<std::string>& GetDecompositionMapping();

} // namespace torch::jit
