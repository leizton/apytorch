#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void ScalarTypeAnalysisForONNX(
    const std::shared_ptr<Graph>& graph,
    bool lowprecision_cast,
    int opset_version);
void ScalarTypeAnalysisNodeForONNX(Node* n);

} // namespace jit
} // namespace torch
