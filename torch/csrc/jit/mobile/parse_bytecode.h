#pragma once
#include <torch/csrc/jit/mobile/function.h>

namespace torch {
namespace jit {
namespace mobile {
using c10::IValue;
void parseInstructions(
    const std::string& function_name,
    c10::ivalue::TupleElements&& ins_list,
    c10::ivalue::TupleElements& debug_handles_m_tuple,
    mobile::Function* function);
void parseConstants(
    const c10::ivalue::TupleElements& consts_list,
    mobile::Function* function);
void parseTypes(
    const c10::ivalue::TupleElements& types_list,
    mobile::Function* function);
void parseRegisterSize(size_t rsize, mobile::Function* function);
void applyUpgrader(
    mobile::Function* function,
    uint64_t operator_version);
} // namespace mobile
} // namespace jit
} // namespace torch
