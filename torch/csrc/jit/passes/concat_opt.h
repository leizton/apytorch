#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Eliminates common inputs among `aten::cat` ops.
bool EliminateConcatCommonInputs(const std::shared_ptr<Graph>& graph);

// Expands `aten::cat` ops into `aten::copy` ops and eliminates redudancies
// in the buffers used for concatenation if possible.
void ExpandConcatAndEliminateRedundancy(
    const std::shared_ptr<Graph>& graph);

bool CombineConcats(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
