
#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void CheckStrictFusion(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
