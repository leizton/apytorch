#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// return true if graph is modified
bool RefineIntegerValues(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
