#pragma once
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/named_value.h>

#include <ATen/core/function_schema.h>

namespace torch {
namespace jit {

// Try to match a list of inputs and keyword 'attributes' to this
// schema. Return the flat list of positional inputs to the call or
// `c10::nullopt` on failure (`failure_messages` contains a good error
// report in this case)

struct MatchedSchema {
  std::vector<Value*> inputs;
  std::vector<TypePtr> return_types;
  c10::OptNameList return_field_names;
  std::string schema_name;
};

bool isBlockListedSchema(const FunctionSchema& schema);

MatchedSchema matchSchema(
    const ::c10::FunctionSchema& schema,
    const SourceRange& loc,
    Graph& graph,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self = c10::nullopt);

std::pair<size_t, MatchedSchema> matchSchemas(
    const std::vector<const ::c10::FunctionSchema*>& schemas,
    const SourceRange& loc,
    Graph& graph,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self = c10::nullopt,
    bool render_errors = false);

bool convertibleToList(
    const TypePtr& type,
    const TypePtr& list_type_);

std::string getFullSchemaName(const ::c10::FunctionSchema& schema);

Value* emitBuiltinCall(
    const SourceRange& loc,
    Graph& graph,
    Symbol name,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self = c10::nullopt);

c10::optional<size_t> findInputWithName(
    const std::string& name,
    at::ArrayRef<NamedValue> kwargs,
    bool is_aten = false);

// applies implicit conversion from value trying to turn it into type
// concrete_type it succeeds if the return_value->isSubtypeOf(concrete_type)
Value* tryConvertToType(
    const SourceRange& loc,
    Graph& graph,
    const TypePtr& concrete_type,
    Value* value,
    bool allow_conversions);
} // namespace jit
} // namespace torch
