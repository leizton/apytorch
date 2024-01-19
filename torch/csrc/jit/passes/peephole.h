#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// return true if graph is modified
bool PeepholeOptimize(
    const std::shared_ptr<Graph>& graph,
    bool disable_shape_peepholes = false);
// return true if graph is modified
bool PeepholeOptimize(
    Block* block,
    bool disable_shape_peepholes = false);
// return true if graph is modified
bool FuseAddMM(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
