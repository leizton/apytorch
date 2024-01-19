#pragma once

#include <torch/csrc/Export.h>

namespace torch {
bool get_cpp_stacktraces_enabled();
bool get_disable_addr2line();
} // namespace torch
