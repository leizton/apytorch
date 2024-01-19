#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// updates the types of tuples according to the type of their current inputs.
void RefineTupleTypes(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
