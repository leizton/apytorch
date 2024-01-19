#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

bool EliminateCommonSubexpression(
    const std::shared_ptr<Graph>& graph);
}
} // namespace torch
