#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void RemoveInplaceOpsForONNX(
    const std::shared_ptr<Graph>& graph,
    Module* model);

} // namespace jit
} // namespace torch
