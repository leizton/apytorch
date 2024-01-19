#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include <memory>

namespace torch {
namespace jit {
// see .cpp for docs
void RemoveInplaceOps(const std::shared_ptr<Graph>& graph);

void ImplicitCastForBinaryInplaceOps(Block* block);
} // namespace jit
} // namespace torch
