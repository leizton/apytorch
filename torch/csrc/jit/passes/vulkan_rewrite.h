#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/mobile_optimizer_type.h>

namespace torch {
namespace jit {
void vulkanInsertPrePackedOps(std::shared_ptr<Graph>& graph);
void vulkanInsertPrePackedOps(script::Module& module);
void vulkanFusePrePackedConvWithClamp(script::Module& module);
void vulkanFoldPrePackingOps(script::Module& module);
script::Module vulkanOptimizeForMobile(
    const script::Module& module,
    const std::set<MobileOptimizerType>& optimization_blocklist,
    const std::vector<std::string>& preserved_methods);
} // namespace jit
} // namespace torch
