#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void liftClosures(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
