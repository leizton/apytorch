#pragma once

#include <c10/macros/Macros.h>

namespace torch {
namespace jit {

double strtod_c(const char* nptr, char** endptr);
float strtof_c(const char* nptr, char** endptr);

} // namespace jit
} // namespace torch
