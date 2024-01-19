#pragma once

#include <torch/library.h>

namespace torch {
namespace autograd {

// Default DispatchKey::Autograd fallback for built-in operators.
// Can be registered for custom operators.
torch::CppFunction autogradNotImplementedFallback();

// Default DispatchKey::AdInplaceOrView fallback for built-in operators
// Can be registered for custom operators.
torch::CppFunction autogradNotImplementedInplaceOrViewFallback();

// Default DispatchKey::Autograd fallback for all other operators (i.e. custom
// operators)
torch::CppFunction basicAutogradNotImplementedFallback();

enum class AutogradFallbackMode {
  Nothing, // Fallback is a redispatch
  Warn, // Fallback raises a warning if backward is called
  Error, // Fallback raises an error if backward is called
};

// Change the behavior of "basicAutogradNotImplementedFallback"
// In Python this is:
// - torch._C._set_autograd_fallback_mode(str) -> None
// - torch._C._get_autograd_fallback_mode() -> str
void setAutogradFallbackMode(AutogradFallbackMode mode);
AutogradFallbackMode getAutogradFallbackMode();

} // namespace autograd
} // namespace torch
