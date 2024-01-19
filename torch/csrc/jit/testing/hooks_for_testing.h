#pragma once
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <functional>
#include <memory>

namespace torch {
namespace jit {
struct Module;

using ModuleHook = std::function<void(Module module)>;
using FunctionHook = std::function<void(StrongFunctionPtr function)>;

void didFinishEmitModule(Module module);
void didFinishEmitFunction(StrongFunctionPtr defined);
void setEmitHooks(ModuleHook for_module, FunctionHook for_fn);

std::pair<ModuleHook, FunctionHook> getEmitHooks();

} // namespace jit
} // namespace torch
