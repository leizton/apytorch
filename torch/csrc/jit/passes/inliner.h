#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Inline function and method calls.
void Inline(Graph& graph);

GraphFunction* tryToGraphFunction(Node* n);

} // namespace jit
} // namespace torch
