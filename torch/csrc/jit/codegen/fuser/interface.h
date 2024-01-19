#pragma once

#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace torch {
namespace jit {

constexpr int kCPUDevice = -1;

// Assigns a "key" to the given fusion_group that it can use to run its
// fusion later (via runFusion() below).
int64_t registerFusion(const Node* fusion_group);

// Runs the fusion corresponding to the given key on the inputs
// found on the stack. Outputs are placed on the same stack.
// In some cases a fusion cannot be run and a fallback path where
// PyTorch's interpreter runs the graph instead is attempted.
void runFusion(const int64_t key, Stack& stack);

// True if the respective devices can fuse, false otherwise
bool canFuseOnCPU();
bool canFuseOnGPU();

// Sets whether fusion on the CPU is allowed (disabled by default due to
// flakiness)
void overrideCanFuseOnCPU(bool value);

// Sets whether fusion on CPU must use LLVM Codegen and not SimplieIREval
void overrideMustUseLLVMOnCPU(bool value);

// Sets whether fusion on the GPU is allowed (enabled by default)
void overrideCanFuseOnGPU(bool value);

// Treats the given graph as a fusion group and launches it on the
// specified device with the given inputs.
// Returns the outputs.
std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs);

// Treats the given graph as a fusion group and returns the generated code.
std::string debugGetFusedKernelCode(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs);

size_t nCompiledKernels();

} // namespace jit
} // namespace torch
