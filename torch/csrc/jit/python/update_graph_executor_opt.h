#pragma once
#include <torch/csrc/Export.h>
namespace torch::jit {
void setGraphExecutorOptimize(bool o);
bool getGraphExecutorOptimize();
} // namespace torch::jit
