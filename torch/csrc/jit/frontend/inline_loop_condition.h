#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void InlineLoopCondition(std::shared_ptr<Graph>& graph);
void InlineBlockBeforeNode(Node* before_node, Block* block);

} // namespace jit
} // namespace torch
