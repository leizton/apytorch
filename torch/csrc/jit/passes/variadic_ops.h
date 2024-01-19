#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Try to replace an op that takes a list input with another op that takes a
// variadic number of arguments.
bool UseVariadicOp(
    const std::shared_ptr<Graph>& graph,
    NodeKind op,
    NodeKind variadic_op);

bool RemoveListMutationAndUseVariadicOp(
    const std::shared_ptr<Graph>& graph,
    NodeKind op,
    NodeKind variadic_op);

// Convenient functions for replacing aten::stack/aten::cat with their
// variadic versions.
bool UseVariadicCat(const std::shared_ptr<Graph>& graph);
bool RemoveListMutationAndUseVariadicCat(
    const std::shared_ptr<Graph>& graph);

bool UseVariadicStack(const std::shared_ptr<Graph>& graph);
bool RemoveListMutationAndUseVariadicStack(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
