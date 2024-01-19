#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void unprofileGraphInputs(const std::shared_ptr<Graph>& graph);
void unprofileBlock(Block* start_block);
// Unprofiles all the node outputs in a block.

void ClearProfilingInformation(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
