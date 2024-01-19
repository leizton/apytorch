#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

const std::string& GetSerializedFuncs();

const OperatorMap<std::string>& GetFuncMapping();

} // namespace torch::jit
