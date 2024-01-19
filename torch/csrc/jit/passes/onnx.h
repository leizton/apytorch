#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/onnx/onnx.h>
#include <unordered_map>

namespace torch {
namespace jit {

std::shared_ptr<Graph> ToONNX(
    std::shared_ptr<Graph>& state,
    ::torch::onnx::OperatorExportTypes operator_export_type);
std::unordered_map<Value*, Value*> BlockToONNX(
    Block* old_block,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    std::unordered_map<Value*, Value*>& env,
    bool is_sub_block = false);
void NodeToONNX(
    Node* old_node,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    std::unordered_map<Value*, Value*>& env);
void RemovePrintOps(std::shared_ptr<Graph>& graph);
void PreprocessCaffe2Ops(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
