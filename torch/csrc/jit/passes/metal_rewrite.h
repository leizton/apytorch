#pragma once
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <string>
#include <vector>

namespace torch {
namespace jit {
void metalInsertPrePackedOps(std::shared_ptr<Graph>& graph);
void metalInsertPrePackedOps(script::Module& module);
void metalFusePrePackedConvWithClamp(script::Module& module);
void metalFoldPrePackingOps(script::Module& module);
script::Module metalOptimizeForMobile(
    const script::Module& module,
    const std::vector<std::string>& preserved_methods);
} // namespace jit
} // namespace torch
