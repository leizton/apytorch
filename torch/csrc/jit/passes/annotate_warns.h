#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void AnnotateWarns(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
