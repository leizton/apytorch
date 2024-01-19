#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Utility functions for PyTorch to ONNX conversion.

static const int OPSET_VERSION_1 = 1;
static const int OPSET_VERSION_9 = 9;
static const int OPSET_VERSION_10 = 10;
static const int OPSET_VERSION_11 = 11;
static const int OPSET_VERSION_12 = 12;
static const int OPSET_VERSION_13 = 13;
static const int OPSET_VERSION_14 = 14;
static const int OPSET_VERSION_15 = 15;
static const int OPSET_VERSION_16 = 16;

using ValueToParamPairMap = std::map<Value*, std::pair<std::string, IValue>>;

using ParamMap = std::map<std::string, IValue>;

void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict);
ValueToParamPairMap
buildValueToParamsMap(Block* b, const ParamMap& paramsDict);
void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap);
void eraseUnusedBlockInputs(Block* b);
void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict);

Node* addNodeToBlock(
    Block* block,
    Symbol kind,
    ArrayRef<Value*> inputs);

Value* addInputToBlock(Block* block);

c10::optional<at::ScalarType> ONNXTypeToATenType(int32_t onnx_type);

// Use int return type as no sable way exists to forward declare protobuf enum
int ATenTypeToOnnxType(at::ScalarType at_type);

void ONNXLintGraph(const std::shared_ptr<Graph>& graph);

Node* createONNXUnsqueeze(
    Graph* graph,
    Node* n_to_insert_before,
    Value* input,
    int axis,
    int opset_version);
Node* createONNXConstant(
    Graph* graph,
    Node* n_to_insert_before,
    at::Tensor value);

bool isValidToTransformToONNXConcatNode(Node* lc_node);

Node* transformToONNXConcatNode(
    Graph* graph,
    Node* lc_node,
    bool need_new_input,
    int opset_version);

class ScalarTypeHashFunction {
 public:
  size_t operator()(const c10::ScalarType& type) const {
    return static_cast<size_t>(type);
  }
};

} // namespace jit
} // namespace torch
