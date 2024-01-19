#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

bool AddIfThenElseOp(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
