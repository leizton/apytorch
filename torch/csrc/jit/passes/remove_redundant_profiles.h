#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void RemoveRedundantProfiles(std::shared_ptr<Graph>& graph);
void RemoveRedundantProfiles(Block* block, AliasDb& db);
} // namespace jit
} // namespace torch
