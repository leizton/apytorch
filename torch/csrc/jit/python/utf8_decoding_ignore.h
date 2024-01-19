#pragma once
#include <torch/csrc/Export.h>
namespace torch::jit {
void setUTF8DecodingIgnore(bool o);
bool getUTF8DecodingIgnore();
} // namespace torch::jit
