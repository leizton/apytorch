#pragma once

#include <ATen/core/function_schema.h>
#include <c10/macros/Macros.h>
#include <string>
#include <variant>

namespace torch {
namespace jit {

std::variant<c10::OperatorName, c10::FunctionSchema> parseSchemaOrName(
    const std::string& schemaOrName);
c10::FunctionSchema parseSchema(const std::string& schema);
c10::OperatorName parseName(const std::string& name);

} // namespace jit
} // namespace torch
