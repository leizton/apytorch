#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// removes tuples where TupleConstruct and TupleUnpack are matched
// but leaves tuples in place across if statements, loops, and as inputs/outputs
void LowerSimpleTuples(const std::shared_ptr<Graph>& graph);

// removes _all_ tuples and raises an error if some cannot be removed
// this is used by ONNX to ensure there are not tuples before conversion,
// but will not work on graphs whose inputs contain tuples.
void LowerAllTuples(const std::shared_ptr<Graph>& graph);

void LowerSimpleTuples(Block* block);

} // namespace jit
} // namespace torch
