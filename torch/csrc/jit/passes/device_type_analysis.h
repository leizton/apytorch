#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
struct Graph;

// Propagates Device type info throughout the given graph.
bool DeviceTypePropagation(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
