
#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void Autocast(const std::shared_ptr<Graph>& graph);

bool setAutocastMode(bool value);
bool autocastEnabled();

} // namespace jit
} // namespace torch
