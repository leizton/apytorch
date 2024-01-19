#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void removeDropout(std::shared_ptr<Graph>& graph);

void removeDropout(script::Module& module);

} // namespace jit
} // namespace torch
