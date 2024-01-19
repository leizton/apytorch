#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are
// merged. Ideally this should all go into native_functions.yaml

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

c10::optional<std::shared_ptr<Graph>> GetDecomposition(
    const FunctionSchema& schema);

void RegisterDecomposition(
    const FunctionSchema& schema,
    std::shared_ptr<Graph> g);

void RunDecompositions(std::shared_ptr<Graph> g);

c10::optional<GraphFunction*> GetDecompositionFunction(
    const FunctionSchema& schema);

// For invocation in C++, recommended is to assign to static local variable
Function* GetDecompositionExecutor(const char* schema_literal);

Function* GetDecompositionExecutor(const FunctionSchema& schema);

void run_jit_decomposition(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack);

bool has_jit_decomposition(const FunctionSchema& schema);

} // namespace torch::jit
