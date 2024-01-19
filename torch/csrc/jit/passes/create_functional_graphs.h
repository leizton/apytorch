#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void CreateFunctionalGraphs(const std::shared_ptr<Graph>& graph);

void InlineFunctionalGraphs(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
