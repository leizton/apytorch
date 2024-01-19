#pragma once
#include <torch/csrc/Export.h>
#include <memory>
#include <ostream>
#include <string>

namespace torch {
namespace jit {
namespace onnx {

bool is_log_enabled();

void set_log_enabled(bool enabled);

void set_log_output_stream(std::shared_ptr<std::ostream> out_stream);

std::ostream& _get_log_output_stream();

#define ONNX_LOG(...)                            \
  if (::torch::jit::onnx::is_log_enabled()) {    \
    ::torch::jit::onnx::_get_log_output_stream() \
        << ::c10::str(__VA_ARGS__) << std::endl; \
  }

} // namespace onnx
} // namespace jit
} // namespace torch
