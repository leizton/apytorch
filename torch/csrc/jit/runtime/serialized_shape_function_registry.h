#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

const std::string& GetSerializedShapeFunctions();

const OperatorMap<std::string>& GetShapeFunctionMappings();

const OperatorMap<std::pair<std::string, std::string>>&
GetBoundedShapeMappings();

} // namespace torch::jit
