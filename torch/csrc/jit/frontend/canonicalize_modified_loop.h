#pragma once
#include <memory>

#include <torch/csrc/Export.h>

namespace torch {
namespace jit {

struct Graph;

// Transforms loops so that they can be represented as python
// for or while loops
void CanonicalizeModifiedLoops(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
