#pragma once

#include <c10/util/irange.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace autograd {

struct Error : public Node {
  Error(std::string msg, edge_list&& next_edges)
      : Node(std::move(next_edges)), msg(std::move(msg)) {}

  Error(std::string msg) : msg(std::move(msg)) {}

  variable_list apply(variable_list&& inputs) override;

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  std::string msg;
};

// We print grad_fn names in tensor printing. For functions with backward
// NYI, grad_fn=<Error> will be printed if we use Error, which is confusing. So
// special case with a new NotImplemented function here.
struct NotImplemented : public Error {
  NotImplemented(const std::string& forward_fn, edge_list&& next_edges)
      : Error(
            "derivative for " + forward_fn + " is not implemented",
            std::move(next_edges)) {}

  NotImplemented(const std::string& forward_fn)
      : Error("derivative for " + forward_fn + " is not implemented") {}
};

// Identity in forward, Error in backward. Used to implement
// @once_differentiable
struct DelayedError : public Node {
  DelayedError(std::string msg, int64_t num_inputs) : msg(std::move(msg)) {
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    for (const auto i : c10::irange(num_inputs)) {
      (void)i; // Suppress unused variable warning
      add_input_metadata(Node::undefined_input());
    }
  }

  variable_list apply(variable_list&& inputs) override;

  std::string msg;
};

struct UndefinedGrad : public Node {
  UndefinedGrad() {
    add_input_metadata(Node::undefined_input());
  }

  variable_list apply(variable_list&& inputs) override;
};

struct UndefinedGradBackward : public Node {
  UndefinedGradBackward(edge_list&& next_edges) : Node(std::move(next_edges)) {}

  UndefinedGradBackward() = default;

  variable_list apply(variable_list&& inputs) override;

  void compiled_args(CompiledNodeArgs& args) override {}
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override {
    return apply(variable_list(inputs));
  }
};

struct GraphRoot : public Node {
  GraphRoot(edge_list functions, variable_list inputs)
      : Node(std::move(functions)), outputs(std::move(inputs)) {
    // Ensures calls to stream() on a GraphRoot instance reflect current
    // stream(s) on devices of root grad tensors at the time the instance is
    // constructed.
    for (const auto& t : outputs) {
      add_input_metadata(t);
    }
  }

  variable_list apply(variable_list&& inputs) override {
    return outputs;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  variable_list outputs;
};

struct Identity : public Node {
  variable_list apply(variable_list&& inputs) override;
};

} // namespace autograd
} // namespace torch
