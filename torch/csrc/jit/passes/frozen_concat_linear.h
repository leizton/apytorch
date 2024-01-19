#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Concats multiple linear ops with the same Tensor input
// into a single linear op.
bool FrozenConcatLinear(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
