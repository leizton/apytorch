
#pragma once

// @generated from ../tools/autograd/templates/Functions.h

#include <ATen/ATen.h>
#include <ATen/TensorGeometry.h>
#include <ATen/core/functional.h>

#include <torch/csrc/Export.h>
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/autograd/variable.h"

#include <c10/core/SymIntArrayRef.h>

namespace torch {
namespace autograd {
namespace generated {

using at::ArrayRef;
using at::IntArrayRef;
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using at::TensorGeometry;
using at::Type;
using c10::fmap;
using c10::optional;

inline std::vector<Tensor> unpack_list(
    at::ArrayRef<SavedVariable> xs,
    std::shared_ptr<Node> saved_for = nullptr) {
  // NB: we must explicitly do the conversion in the lambda, otherwise template
  // deduction will give a Tensor of Variable which is not convertible
  return fmap(xs, [&saved_for](const SavedVariable& x) {
    // TODO(crcrpar): Use `std::move(saved_for)` to avoid incrementing refcount,
    // which would need refactoring.
    return static_cast<Tensor>(x.unpack(saved_for));
  });
}

inline c10::List<c10::optional<Tensor>> unpack_opt_list(
    at::ArrayRef<SavedVariable> xs,
    std::shared_ptr<Node> saved_for = nullptr) {
  torch::List<c10::optional<Tensor>> result;
  result.reserve(xs.size());
  for (const SavedVariable& v : xs) {
    auto var = v.unpack(saved_for);
    result.push_back(var.defined() ? c10::optional<Tensor>(var) : c10::nullopt);
  }
  return result;
}

using torch::autograd::TypeAndSize;

struct AbsBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AbsBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct AcosBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AcosBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct AddBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AddBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::ScalarType other_scalar_type;
  at::ScalarType self_scalar_type;
};
struct AddBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AddBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::ScalarType self_scalar_type;
};
struct AddbmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AddbmmBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    batch1_.reset_data();
    batch2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  SavedVariable batch1_;
  c10::SymInt batch1_sym_argsize_0;
  c10::SymInt batch1_sym_argsize_1;
  SavedVariable batch2_;
  c10::SymInt batch2_sym_argsize_2;
  at::Scalar beta;
};
struct AddcdivBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AddcdivBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    tensor1_.reset_data();
    tensor2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::ScalarType self_scalar_type;
  SavedVariable tensor1_;
  at::ScalarType tensor1_scalar_type;
  SavedVariable tensor2_;
  at::ScalarType tensor2_scalar_type;
  at::Scalar value;
};
struct AddcmulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AddcmulBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    tensor1_.reset_data();
    tensor2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::ScalarType self_scalar_type;
  SavedVariable tensor1_;
  at::ScalarType tensor1_scalar_type;
  SavedVariable tensor2_;
  at::ScalarType tensor2_scalar_type;
  at::Scalar value;
};
struct AddmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AddmmBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat1_.reset_data();
    mat2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable mat1_;
  at::Layout mat1_layout;
  std::vector<c10::SymInt> mat1_sym_sizes;
  std::vector<c10::SymInt> mat1_sym_strides;
  SavedVariable mat2_;
  at::Layout mat2_layout;
  std::vector<c10::SymInt> mat2_sym_sizes;
  std::vector<c10::SymInt> mat2_sym_strides;
};
struct SparseAddmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SparseAddmmBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat1_.reset_data();
    mat2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable mat1_;
  SavedVariable mat2_;
  at::Layout mat2_layout;
  std::vector<c10::SymInt> mat2_sym_sizes;
  std::vector<c10::SymInt> mat2_sym_strides;
};
struct AddmvBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AddmvBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat_.reset_data();
    vec_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable mat_;
  SavedVariable vec_;
};
struct AddrBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AddrBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    vec1_.reset_data();
    vec2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable vec1_;
  SavedVariable vec2_;
};
struct AffineGridGeneratorBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AffineGridGeneratorBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> size;
};
struct AliasBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AliasBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct AngleBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AngleBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct AcoshBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AcoshBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct AcoshBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AcoshBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct AsinhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AsinhBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct AsinhBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AsinhBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct AtanhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AtanhBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct AtanhBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AtanhBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct AsStridedBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AsStridedBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::TensorGeometry self_geometry;
  std::vector<c10::SymInt> size;
  c10::optional<c10::SymInt> storage_offset;
  std::vector<c10::SymInt> stride;
};
struct AsStridedBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AsStridedBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::TensorGeometry self_geometry;
  std::vector<c10::SymInt> size;
  c10::optional<c10::SymInt> storage_offset;
  std::vector<c10::SymInt> stride;
};
struct AsinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AsinBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct AtanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AtanBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct Atan2Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Atan2Backward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct BaddbmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "BaddbmmBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    batch1_.reset_data();
    batch2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  SavedVariable batch1_;
  SavedVariable batch2_;
  at::Scalar beta;
};
struct BernoulliBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "BernoulliBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct BernoulliBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "BernoulliBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize p_info;
};
struct BernoulliBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "BernoulliBackward2";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct BmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "BmmBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat2_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable mat2_;
  SavedVariable self_;
};
struct MatmulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MatmulBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct CatBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CatBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  ::std::vector<at::ScalarType> tensors_args_scalartypes;
  ::std::vector<::std::vector<c10::SymInt>> tensors_args_sizes_symint;
  size_t tensors_size_;
};
struct CauchyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CauchyBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct CeilBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CeilBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct CholeskyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CholeskyBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool upper;
  SavedVariable result_;
};
struct LinalgCholeskyExBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgCholeskyExBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    L_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool upper;
  SavedVariable L_;
};
struct CholeskySolveBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CholeskySolveBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input2_.reset_data();
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable input2_;
  SavedVariable self_;
  bool upper;
  SavedVariable result_;
};
struct CholeskyInverseBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CholeskyInverseBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  bool upper;
  SavedVariable result_;
};
struct ClampBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ClampBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    max_.reset_data();
    min_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable max_;
  SavedVariable min_;
  SavedVariable self_;
};
struct ClampBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ClampBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::optional<at::Scalar> max;
  c10::optional<at::Scalar> min;
  SavedVariable self_;
};
struct ClampMinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ClampMinBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar min;
  SavedVariable self_;
};
struct ClampMinBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ClampMinBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    min_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable min_;
  SavedVariable self_;
};
struct ClampMaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ClampMaxBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar max;
  SavedVariable self_;
};
struct ClampMaxBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ClampMaxBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    max_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable max_;
  SavedVariable self_;
};
struct CloneBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CloneBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct ToCopyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ToCopyBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::TensorOptions self_options;
};
struct CoalesceBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CoalesceBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct ComplexBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ComplexBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    imag_.reset_data();
    real_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable imag_;
  SavedVariable real_;
};
struct PolarBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PolarBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct ConjBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConjBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct NegViewBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NegViewBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct ConjPhysicalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConjPhysicalBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct ConjPhysicalBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConjPhysicalBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct CopysignBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CopysignBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  SavedVariable self_;
  SavedVariable result_;
};
struct CopysignBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CopysignBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct CosBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CosBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct CoshBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CoshBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct LinalgCrossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgCrossBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable other_;
  SavedVariable self_;
};
struct LogcumsumexpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LogcumsumexpBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable result_;
};
struct CumprodBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CumprodBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  at::ScalarType self_scalar_type;
  SavedVariable result_;
};
struct CumsumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CumsumBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::ScalarType self_scalar_type;
};
struct CummaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CummaxBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable indices_;
};
struct CumminBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CumminBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable indices_;
};
struct ConvTbcBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConvTbcBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    bias_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable bias_;
  int64_t pad = 0;
  SavedVariable self_;
  SavedVariable weight_;
};
struct CtcLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CtcLossBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    log_probs_.reset_data();
    targets_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t blank = 0;
  std::vector<int64_t> input_lengths;
  SavedVariable log_probs_;
  std::vector<int64_t> target_lengths;
  SavedVariable targets_;
  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;
};
struct CtcLossBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CtcLossBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_lengths_.reset_data();
    log_probs_.reset_data();
    target_lengths_.reset_data();
    targets_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t blank = 0;
  SavedVariable input_lengths_;
  SavedVariable log_probs_;
  SavedVariable target_lengths_;
  SavedVariable targets_;
  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;
};
struct Deg2RadBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Deg2RadBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct LinalgDetBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgDetBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    A_.reset_data();
    LU_.reset_data();
    pivots_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable A_;
  SavedVariable LU_;
  SavedVariable pivots_;
  SavedVariable result_;
};
struct LinalgSlogdetBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgSlogdetBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    A_.reset_data();
    LU_.reset_data();
    pivots_.reset_data();
    sign_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable A_;
  SavedVariable LU_;
  SavedVariable pivots_;
  SavedVariable sign_;
};
struct BlockDiagBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "BlockDiagBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  ::std::vector<at::ScalarType> tensors_args_scalartypes;
  ::std::vector<::std::vector<int64_t>> tensors_args_sizes;
  size_t tensors_size_;
};
struct DiagEmbedBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "DiagEmbedBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim1 = 0;
  int64_t dim2 = 0;
  int64_t offset = 0;
};
struct DiagonalBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "DiagonalBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim1 = 0;
  int64_t dim2 = 0;
  int64_t offset = 0;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct DiagonalBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "DiagonalBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim1 = 0;
  int64_t dim2 = 0;
  int64_t offset = 0;
};
struct DistBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "DistBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  at::Scalar p;
  SavedVariable self_;
  SavedVariable result_;
};
struct DivBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "DivBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
  at::ScalarType self_scalar_type;
};
struct DivBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "DivBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar other;
  at::ScalarType self_scalar_type;
};
struct DivBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "DivBackward2";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  c10::optional<std::string> rounding_mode;
  SavedVariable self_;
  at::ScalarType self_scalar_type;
};
struct DivBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "DivBackward3";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar other;
  c10::optional<std::string> rounding_mode;
  at::ScalarType self_scalar_type;
};
struct DotBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "DotBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    tensor_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable tensor_;
};
struct VdotBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "VdotBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct FusedDropoutBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FusedDropoutBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double p;
  SavedVariable result1_;
};
struct NativeDropoutBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NativeDropoutBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double p;
  c10::optional<bool> train;
  SavedVariable result1_;
};
struct NativeDropoutBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NativeDropoutBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  SavedVariable mask_;
  double scale;
};
struct EqBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "EqBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;
};
struct EqBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "EqBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;
};
struct ErfBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ErfBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct ErfcBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ErfcBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct SpecialErfcxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialErfcxBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct ErfinvBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ErfinvBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct ExpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ExpBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct Exp2Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Exp2Backward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct Expm1Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Expm1Backward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct ExpandBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ExpandBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct ExponentialBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ExponentialBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct FakeQuantizePerTensorAffineCachemaskBackward0
    : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FakeQuantizePerTensorAffineCachemaskBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable mask_;
};
struct FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0
    : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable mask_;
};
struct FakeQuantizeLearnablePerTensorAffineBackward0
    : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FakeQuantizeLearnablePerTensorAffineBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scale_.reset_data();
    self_.reset_data();
    zero_point_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double grad_factor;
  int64_t quant_max = 0;
  int64_t quant_min = 0;
  SavedVariable scale_;
  SavedVariable self_;
  SavedVariable zero_point_;
};
struct FakeQuantizePerChannelAffineCachemaskBackward0
    : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FakeQuantizePerChannelAffineCachemaskBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable mask_;
};
struct FakeQuantizeLearnablePerChannelAffineBackward0
    : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FakeQuantizeLearnablePerChannelAffineBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scale_.reset_data();
    self_.reset_data();
    zero_point_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t axis = 0;
  double grad_factor;
  int64_t quant_max = 0;
  int64_t quant_min = 0;
  SavedVariable scale_;
  SavedVariable self_;
  SavedVariable zero_point_;
};
struct FusedMovingAvgObsFqHelperBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FusedMovingAvgObsFqHelperBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable mask_;
};
struct FillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FillBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct FillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FillBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct FillBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FillBackward2";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct FillBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FillBackward3";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct FloorBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FloorBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct FmodBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FmodBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct FmodBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FmodBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct FracBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FracBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct FrexpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FrexpBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable exponent_;
};
struct GatherBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GatherBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;
  SavedVariable self_;
  bool sparse_grad;
};
struct GeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GeBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;
};
struct GeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GeBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;
};
struct GeometricBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GeometricBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct GeqrfBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GeqrfBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct GridSampler2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GridSampler2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grid_.reset_data();
    input_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  SavedVariable grid_;
  SavedVariable input_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;
};
struct GridSampler3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GridSampler3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grid_.reset_data();
    input_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  SavedVariable grid_;
  SavedVariable input_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;
};
struct GridSampler2DCpuFallbackBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GridSampler2DCpuFallbackBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grid_.reset_data();
    input_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  SavedVariable grid_;
  SavedVariable input_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;
};
struct GtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GtBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;
};
struct GtBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GtBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;
};
struct HardsigmoidBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "HardsigmoidBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct HardswishBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "HardswishBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct HardswishBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "HardswishBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  SavedVariable self_;
  at::TensorOptions self_options;
};
struct HypotBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "HypotBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
  SavedVariable result_;
};
struct I0Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "I0Backward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct SpecialI0EBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialI0EBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct SpecialI1Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialI1Backward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct SpecialI1EBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialI1EBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct IgammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "IgammaBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct IgammacBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "IgammacBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct IndexBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "IndexBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct UnsafeIndexBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnsafeIndexBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct IndexAddBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "IndexAddBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    source_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  int64_t dim = 0;
  SavedVariable index_;
  SavedVariable source_;
  int64_t source_dim = 0;
};
struct IndexReduceBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "IndexReduceBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    self_.reset_data();
    source_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool include_self;
  SavedVariable index_;
  std::string reduce;
  SavedVariable self_;
  SavedVariable source_;
  SavedVariable result_;
};
struct IndexCopyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "IndexCopyBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    source_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;
  SavedVariable source_;
  int64_t source_dim = 0;
};
struct IndexFillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "IndexFillBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;
};
struct IndexFillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "IndexFillBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;
};
struct IndexPutBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "IndexPutBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool accumulate;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  torch::autograd::generated::TypeAndSize values_info;
};
struct UnsafeIndexPutBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnsafeIndexPutBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool accumulate;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  torch::autograd::generated::TypeAndSize values_info;
};
struct IndexPutImplBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "IndexPutImplBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.clear();
    indices_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool accumulate;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  torch::autograd::generated::TypeAndSize values_info;
};
struct IndexSelectBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "IndexSelectBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct LinalgInvExBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgInvExBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    inverse_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable inverse_;
};
struct LinalgPinvBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgPinvBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct KthvalueBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "KthvalueBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;
};
struct LeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LeBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;
};
struct LeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LeBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;
};
struct LerpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LerpBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar weight;
};
struct LerpBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LerpBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    end_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable end_;
  SavedVariable self_;
  SavedVariable weight_;
};
struct LgammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LgammaBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct DigammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "DigammaBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct PolygammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PolygammaBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t n = 0;
  SavedVariable self_;
};
struct PolygammaBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PolygammaBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t n = 0;
  SavedVariable self_;
};
struct LogBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LogBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct Log10Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Log10Backward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct Log1PBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Log1PBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct Log2Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Log2Backward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct LogaddexpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LogaddexpBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct Logaddexp2Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Logaddexp2Backward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct XlogyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "XlogyBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct XlogyBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "XlogyBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  at::Scalar self;
};
struct XlogyBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "XlogyBackward2";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar other;
  SavedVariable self_;
};
struct SpecialXlog1PyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialXlog1PyBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct SpecialXlog1PyBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialXlog1PyBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  at::Scalar self;
};
struct SpecialXlog1PyBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialXlog1PyBackward2";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar other;
  SavedVariable self_;
};
struct SpecialZetaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialZetaBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct SpecialZetaBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialZetaBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  at::Scalar self;
};
struct SpecialZetaBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialZetaBackward2";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct LogNormalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LogNormalBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct LogsumexpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LogsumexpBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
  SavedVariable result_;
};
struct LinalgLstsqBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgLstsqBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    b_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable b_;
  SavedVariable self_;
};
struct LtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LtBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;
};
struct LtBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LtBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;
};
struct LinalgLuFactorExBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgLuFactorExBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    LU_.reset_data();
    pivots_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool pivot;
  SavedVariable LU_;
  SavedVariable pivots_;
};
struct LinalgLuBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgLuBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    L_.reset_data();
    P_.reset_data();
    U_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool pivot;
  SavedVariable L_;
  SavedVariable P_;
  SavedVariable U_;
};
struct LinalgLuSolveBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgLuSolveBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    LU_.reset_data();
    pivots_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable LU_;
  bool adjoint;
  bool left;
  SavedVariable pivots_;
  SavedVariable result_;
};
struct LuUnpackBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LuUnpackBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::SymInt LU_data_sym_argsize_minus_1;
  c10::SymInt LU_data_sym_argsize_minus_2;
};
struct MaskedFillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaskedFillBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable mask_;
};
struct MaskedFillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaskedFillBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable mask_;
};
struct MaskedScatterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaskedScatterBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable mask_;
  std::vector<c10::SymInt> source_sym_sizes;
};
struct MaskedScatterBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaskedScatterBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize grad_output_info;
  SavedVariable mask_;
};
struct MaskedSelectBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaskedSelectBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable mask_;
  SavedVariable self_;
};
struct LinalgMatrixExpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgMatrixExpBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct MaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaxBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;
};
struct MaxBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaxBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct MaximumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaximumBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct FmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FmaxBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct MeanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MeanBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::SymInt self_sym_numel;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct MeanBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MeanBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  c10::SymInt self_sym_numel;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct MedianBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MedianBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct NanmedianBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NanmedianBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct MedianBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MedianBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;
};
struct NanmedianBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NanmedianBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;
};
struct MinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MinBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;
};
struct MinBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MinBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct MinimumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MinimumBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct FminBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FminBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct AmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AmaxBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
  SavedVariable result_;
};
struct AminBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AminBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
  SavedVariable result_;
};
struct MmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MmBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat2_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable mat2_;
  at::Layout mat2_layout;
  std::vector<c10::SymInt> mat2_sym_sizes;
  std::vector<c10::SymInt> mat2_sym_strides;
  SavedVariable self_;
  at::Layout self_layout;
  std::vector<c10::SymInt> self_sym_sizes;
  std::vector<c10::SymInt> self_sym_strides;
};
struct ModeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ModeBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;
};
struct MulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MulBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  at::ScalarType other_scalar_type;
  SavedVariable self_;
  at::ScalarType self_scalar_type;
};
struct MulBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MulBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar other;
  at::ScalarType self_scalar_type;
};
struct MvBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MvBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    vec_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable vec_;
};
struct MvlgammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MvlgammaBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t p = 0;
  SavedVariable self_;
};
struct NanToNumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NanToNumBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct NativeBatchNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NativeBatchNormBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double eps;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;
};
struct NativeBatchNormLegitBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NativeBatchNormLegitBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double eps;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;
};
struct NativeBatchNormLegitNoTrainingBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NativeBatchNormLegitNoTrainingBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double eps;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;
};
struct NativeBatchNormLegitBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NativeBatchNormLegitBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double eps;
  SavedVariable input_;
  bool training;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;
};
struct NativeBatchNormBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NativeBatchNormBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_out_.reset_data();
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    save_invstd_.reset_data();
    save_mean_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double eps;
  SavedVariable grad_out_;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_invstd_;
  SavedVariable save_mean_;
  bool train;
  SavedVariable weight_;
};
struct NativeLayerNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NativeLayerNormBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    bias_.reset_data();
    input_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable bias_;
  SavedVariable input_;
  std::vector<c10::SymInt> normalized_shape;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;
};
struct NativeLayerNormBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NativeLayerNormBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_out_.reset_data();
    input_.reset_data();
    mean_.reset_data();
    rstd_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable grad_out_;
  SavedVariable input_;
  SavedVariable mean_;
  std::vector<c10::SymInt> normalized_shape;
  SavedVariable rstd_;
  SavedVariable weight_;
};
struct NativeGroupNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NativeGroupNormBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::SymInt C;
  c10::SymInt HxW;
  c10::SymInt N;
  double eps;
  int64_t group = 0;
  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;
};
struct NeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NeBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;
};
struct NeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NeBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize other_info;
  torch::autograd::generated::TypeAndSize self_info;
};
struct NegBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NegBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct NextafterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NextafterBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct NormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NormBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar p;
  SavedVariable self_;
  SavedVariable result_;
};
struct NormBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NormBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  bool keepdim;
  c10::optional<at::Scalar> p;
  SavedVariable self_;
  SavedVariable result_;
};
struct NormBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NormBackward2";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::optional<at::Scalar> p;
  SavedVariable self_;
  SavedVariable result_;
};
struct NormBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NormBackward3";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  bool keepdim;
  c10::optional<at::Scalar> p;
  SavedVariable self_;
  SavedVariable result_;
};
struct LinalgVectorNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgVectorNormBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  at::Scalar ord;
  SavedVariable self_;
  SavedVariable result_;
};
struct PdistBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PdistBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double p;
  SavedVariable self_;
  SavedVariable result_;
};
struct PdistBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PdistBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct EuclideanDistBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "EuclideanDistBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    x1_.reset_data();
    x2_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable x1_;
  SavedVariable x2_;
  SavedVariable result_;
};
struct CdistBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CdistBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    x1_.reset_data();
    x2_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double p;
  SavedVariable x1_;
  SavedVariable x2_;
  SavedVariable result_;
};
struct CdistBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CdistBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct NormalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NormalBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct NormalBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NormalBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> mean_sym_sizes;
};
struct NormalBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NormalBackward2";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> std_sym_sizes;
};
struct NormalBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NormalBackward3";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> mean_sym_sizes;
  std::vector<c10::SymInt> std_sym_sizes;
};
struct LinalgHouseholderProductBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgHouseholderProductBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    tau_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable input_;
  SavedVariable tau_;
  SavedVariable result_;
};
struct OrmqrBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "OrmqrBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input2_.reset_data();
    input3_.reset_data();
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable input2_;
  SavedVariable input3_;
  bool left;
  SavedVariable self_;
  bool transpose;
  SavedVariable result_;
};
struct PermuteBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PermuteBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dims;
};
struct PoissonBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PoissonBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;
};
struct PowBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PowBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar exponent;
  SavedVariable self_;
};
struct PowBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PowBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.reset_data();
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable exponent_;
  SavedVariable self_;
  SavedVariable result_;
};
struct PowBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PowBackward2";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable exponent_;
  at::Scalar self;
  SavedVariable result_;
};
struct ProdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ProdBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct ProdBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ProdBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable self_;
  SavedVariable result_;
};
struct PutBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PutBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    source_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool accumulate;
  SavedVariable index_;
  SavedVariable source_;
  torch::autograd::generated::TypeAndSize source_info;
};
struct LinalgQrBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgQrBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    Q_.reset_data();
    R_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::string mode;
  SavedVariable Q_;
  SavedVariable R_;
};
struct Rad2DegBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Rad2DegBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct RandomBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RandomBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct RandomBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RandomBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct RandomBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RandomBackward2";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct ReciprocalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReciprocalBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct RemainderBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RemainderBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct RemainderBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RemainderBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct RenormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RenormBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::Scalar maxnorm;
  at::Scalar p;
  SavedVariable self_;
};
struct RepeatBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RepeatBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> repeats;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SpecialEntrBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialEntrBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct SpecialNdtriBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialNdtriBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct SpecialLogNdtrBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SpecialLogNdtrBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct ReshapeAliasBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReshapeAliasBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct RoundBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RoundBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct RoundBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RoundBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct RsqrtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RsqrtBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct ScatterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ScatterBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;
};
struct ScatterBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ScatterBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;
};
struct ScatterAddBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ScatterAddBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable index_;
};
struct SelectBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SelectBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt index;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SelectBackwardAutogradNestedTensor0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SelectBackwardAutogradNestedTensor0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt index;
  SavedVariable self_;
};
struct SelectBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SelectBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt index;
};
struct SigmoidBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SigmoidBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct LogitBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LogitBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::optional<double> eps;
  SavedVariable self_;
};
struct SignBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SignBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct SgnBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SgnBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct SinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SinBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct SincBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SincBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct SinhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SinhBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct SliceBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SliceBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::optional<c10::SymInt> end;
  std::vector<c10::SymInt> self_sym_sizes;
  c10::optional<c10::SymInt> start;
  c10::SymInt step;
};
struct SliceBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SliceBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt end;
  c10::SymInt start;
  c10::SymInt step;
};
struct SliceScatterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SliceScatterBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::optional<c10::SymInt> end;
  torch::autograd::generated::TypeAndSize src_info;
  c10::optional<c10::SymInt> start;
  c10::SymInt step;
};
struct SelectScatterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SelectScatterBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt index;
  torch::autograd::generated::TypeAndSize src_info;
};
struct DiagonalScatterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "DiagonalScatterBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim1 = 0;
  int64_t dim2 = 0;
  int64_t offset = 0;
  torch::autograd::generated::TypeAndSize src_info;
};
struct AsStridedScatterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AsStridedScatterBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::TensorGeometry self_geometry;
  std::vector<c10::SymInt> size;
  at::TensorGeometry src_geometry;
  c10::optional<c10::SymInt> storage_offset;
  std::vector<c10::SymInt> stride;
};
struct LinalgSolveExBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgSolveExBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    A_.reset_data();
    LU_.reset_data();
    pivots_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable A_;
  bool left;
  SavedVariable LU_;
  SavedVariable pivots_;
  SavedVariable result_;
};
struct SortBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SortBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;
};
struct SortBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SortBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;
};
struct SplitBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SplitBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
  c10::SymInt split_size;
};
struct UnsafeSplitBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnsafeSplitBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
  c10::SymInt split_size;
};
struct SplitWithSizesBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SplitWithSizesBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
  std::vector<c10::SymInt> split_sizes;
};
struct SplitWithSizesBackwardAutogradNestedTensor0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SplitWithSizesBackwardAutogradNestedTensor0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> split_sizes;
};
struct UnsafeSplitWithSizesBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnsafeSplitWithSizesBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
  std::vector<c10::SymInt> split_sizes;
};
struct SqrtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqrtBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct SqueezeBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SqueezeBackward1 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SqueezeBackwardAutogradNestedTensor0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackwardAutogradNestedTensor0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
};
struct SqueezeBackward2 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackward2";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SqueezeBackwardAutogradNestedTensor1 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackwardAutogradNestedTensor1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  int64_t self_dim = 0;
};
struct SqueezeBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackward3";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SqueezeBackward4 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackward4";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SqueezeBackward5 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackward5";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct StdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "StdBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::optional<at::Scalar> correction;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
  SavedVariable result_;
};
struct StdMeanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "StdMeanBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result0_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::optional<at::Scalar> correction;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
  SavedVariable result0_;
};
struct SubBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SubBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::ScalarType other_scalar_type;
  at::ScalarType self_scalar_type;
};
struct SubBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SubBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::ScalarType self_scalar_type;
};
struct RsubBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RsubBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::ScalarType other_scalar_type;
  at::ScalarType self_scalar_type;
};
struct RsubBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RsubBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::ScalarType self_scalar_type;
};
struct SumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SumBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SumBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SumBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SumBackwardAutogradNestedTensor0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SumBackwardAutogradNestedTensor0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
};
struct NansumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NansumBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
  at::ScalarType self_scalar_type;
};
struct LinalgSvdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgSvdBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    S_.reset_data();
    U_.reset_data();
    Vh_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool full_matrices;
  SavedVariable S_;
  c10::SymInt S_sym_argsize_minus_1;
  SavedVariable U_;
  SavedVariable Vh_;
};
struct LinalgEighBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgEighBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    eigenvalues_.reset_data();
    eigenvectors_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_;
};
struct LinalgEigBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgEigBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    eigenvalues_.reset_data();
    eigenvectors_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::ScalarType self_scalar_type;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_;
};
struct TBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct TBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct FlipBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FlipBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dims;
};
struct RollBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RollBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dims;
  std::vector<c10::SymInt> shifts;
};
struct Rot90Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Rot90Backward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dims;
  int64_t k = 0;
};
struct TakeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TakeBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable index_;
  SavedVariable self_;
};
struct TanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TanBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct TanhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TanhBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct TopkBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TopkBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  std::vector<c10::SymInt> self_sym_sizes;
  SavedVariable indices_;
};
struct TraceBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TraceBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct TransposeBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TransposeBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim0 = 0;
  int64_t dim1 = 0;
};
struct TransposeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TransposeBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim0 = 0;
  int64_t dim1 = 0;
};
struct TriangularSolveBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TriangularSolveBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    A_.reset_data();
    self_.reset_data();
    solution_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable A_;
  SavedVariable self_;
  bool transpose;
  bool unitriangular;
  bool upper;
  SavedVariable solution_;
};
struct LinalgSolveTriangularBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinalgSolveTriangularBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool left;
  SavedVariable self_;
  bool unitriangular;
  bool upper;
  SavedVariable result_;
};
struct TrilBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TrilBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t diagonal = 0;
};
struct TriuBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TriuBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t diagonal = 0;
};
struct TruncBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TruncBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct ToDenseBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ToDenseBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::optional<bool> masked_grad;
  SavedVariable self_;
};
struct ToSparseBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ToSparseBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Layout self_layout;
  c10::OptionalArray<c10::SymInt> self_self_sym_blocksize_opt;
};
struct ToSparseBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ToSparseBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Layout self_layout;
  c10::OptionalArray<c10::SymInt> self_self_sym_blocksize_opt;
};
struct ToSparseCsrBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ToSparseCsrBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Layout self_layout;
  c10::OptionalArray<c10::SymInt> self_self_sym_blocksize_opt;
};
struct ToSparseCscBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ToSparseCscBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Layout self_layout;
  c10::OptionalArray<c10::SymInt> self_self_sym_blocksize_opt;
};
struct ToSparseBsrBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ToSparseBsrBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Layout self_layout;
  c10::OptionalArray<c10::SymInt> self_self_sym_blocksize_opt;
};
struct ToSparseBscBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ToSparseBscBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Layout self_layout;
  c10::OptionalArray<c10::SymInt> self_self_sym_blocksize_opt;
};
struct ToMkldnnBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ToMkldnnBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct UnfoldBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnfoldBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dimension = 0;
  std::vector<c10::SymInt> self_sym_sizes;
  int64_t size = 0;
  int64_t step = 0;
};
struct UnfoldBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnfoldBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  int64_t size = 0;
  int64_t step = 0;
};
struct UniformBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UniformBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct UniqueBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UniqueBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct UniqueDimBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UniqueDimBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct UniqueConsecutiveBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UniqueConsecutiveBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct UniqueDimConsecutiveBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UniqueDimConsecutiveBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct Unique2Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Unique2Backward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct UnsafeViewBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnsafeViewBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct LiftBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LiftBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct LiftFreshBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LiftFreshBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct UnsqueezeBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnsqueezeBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
};
struct UnsqueezeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnsqueezeBackward1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
};
struct VarBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "VarBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::optional<at::Scalar> correction;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
};
struct VarMeanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "VarMeanBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::optional<at::Scalar> correction;
  c10::OptionalArray<int64_t> dim;
  bool keepdim;
  SavedVariable self_;
};
struct ViewBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ViewBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct ViewBackwardAutogradNestedTensor0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ViewBackwardAutogradNestedTensor0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct ViewAsRealBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ViewAsRealBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct ViewAsComplexBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ViewAsComplexBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct WhereBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "WhereBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    condition_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable condition_;
};
struct WeightNormInterfaceBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "WeightNormInterfaceBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    g_.reset_data();
    v_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable g_;
  SavedVariable v_;
  SavedVariable result1_;
};
struct ZeroBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ZeroBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct SparseMaskBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SparseMaskBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable mask_;
  at::Layout self_layout;
};
struct SparseCooTensorWithDimsAndTensorsBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SparseCooTensorWithDimsAndTensorsBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct SparseCompressedTensorBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SparseCompressedTensorBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    values_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable values_;
  SavedVariable result_;
};
struct SparseSumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SparseSumBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  SavedVariable self_;
};
struct StandardGammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "StandardGammaBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result_;
};
struct StandardGammaGradBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "StandardGammaGradBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct ValuesBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ValuesBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct ValuesBackwardAutogradNestedTensor0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ValuesBackwardAutogradNestedTensor0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct TrilinearBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TrilinearBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    i1_.reset_data();
    i2_.reset_data();
    i3_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> expand1;
  std::vector<int64_t> expand2;
  std::vector<int64_t> expand3;
  SavedVariable i1_;
  SavedVariable i2_;
  SavedVariable i3_;
  std::vector<int64_t> sumdim;
};
struct ConstantPadNdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConstantPadNdBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> pad;
};
struct BinaryCrossEntropyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "BinaryCrossEntropyBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
};
struct BinaryCrossEntropyBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "BinaryCrossEntropyBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
};
struct BinaryCrossEntropyWithLogitsBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "BinaryCrossEntropyWithLogitsBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    pos_weight_.reset_data();
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable pos_weight_;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
};
struct EmbeddingBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "EmbeddingBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable indices_;
  c10::SymInt padding_idx;
  bool scale_grad_by_freq;
  bool sparse;
  c10::SymInt weight_sym_argsize_0;
};
struct EmbeddingDenseBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "EmbeddingDenseBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable indices_;
  c10::SymInt padding_idx;
};
struct EmbeddingBagBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "EmbeddingBagBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
    offsets_.reset_data();
    per_sample_weights_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable indices_;
  int64_t mode = 0;
  SavedVariable offsets_;
  int64_t padding_idx = 0;
  SavedVariable per_sample_weights_;
  bool scale_grad_by_freq;
  bool sparse;
  SavedVariable weight_;
  c10::SymInt weight_sym_argsize_0;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;
};
struct EmbeddingRenormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "EmbeddingRenormBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct MseLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MseLossBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
};
struct MultiMarginLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MultiMarginLossBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar margin;
  at::Scalar p;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
};
struct MultilabelMarginLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MultilabelMarginLossBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    is_target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable is_target_;
};
struct NllLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NllLossBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
    total_weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::SymInt ignore_index;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  SavedVariable total_weight_;
};
struct NllLoss2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NllLoss2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
    total_weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::SymInt ignore_index;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  SavedVariable total_weight_;
};
struct SmoothL1LossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SmoothL1LossBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double beta;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
};
struct HuberLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "HuberLossBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double delta;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
};
struct SoftMarginLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SoftMarginLossBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
};
struct ReluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReluBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable result_;
};
struct SiluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SiluBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct MishBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MishBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct EluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "EluBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar input_scale;
  at::Scalar scale;
  SavedVariable self_;
};
struct EluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "EluBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar input_scale;
  at::Scalar scale;
  SavedVariable result_;
};
struct CeluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CeluBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  SavedVariable self_;
};
struct CeluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CeluBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  SavedVariable result_;
};
struct GeluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GeluBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::string approximate;
  SavedVariable self_;
};
struct GeluBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GeluBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::string approximate;
  SavedVariable grad_output_;
  SavedVariable self_;
};
struct GluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GluBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
};
struct HardshrinkBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "HardshrinkBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar lambd;
  SavedVariable self_;
};
struct HardshrinkBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "HardshrinkBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar lambd;
  SavedVariable self_;
};
struct HardtanhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "HardtanhBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar max_val;
  at::Scalar min_val;
  SavedVariable self_;
};
struct LeakyReluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LeakyReluBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar negative_slope;
  SavedVariable self_;
};
struct LeakyReluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LeakyReluBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar negative_slope;
  SavedVariable result_;
};
struct LogSigmoidBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LogSigmoidBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    buffer_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable buffer_;
};
struct LogSoftmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LogSoftmaxBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::ScalarType self_scalar_type;
  SavedVariable result_;
};
struct SparseLogSoftmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SparseLogSoftmaxBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable result_;
};
struct MaskedSoftmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaskedSoftmaxBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::optional<int64_t> dim;
  SavedVariable mask_;
  SavedVariable result_;
};
struct PreluKernelBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PreluKernelBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable weight_;
};
struct PreluKernelBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PreluKernelBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  at::TensorOptions grad_output_options;
  SavedVariable self_;
  torch::autograd::generated::TypeAndSize self_info;
  at::TensorOptions self_options;
  SavedVariable weight_;
  at::TensorOptions weight_options;
};
struct RreluWithNoiseBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RreluWithNoiseBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    noise_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar lower;
  SavedVariable noise_;
  SavedVariable self_;
  bool training;
  at::Scalar upper;
};
struct RreluWithNoiseBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RreluWithNoiseBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    noise_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar lower;
  SavedVariable noise_;
  bool training;
  at::Scalar upper;
  SavedVariable result_;
};
struct SoftmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SoftmaxBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::ScalarType self_scalar_type;
  SavedVariable result_;
};
struct SparseSoftmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SparseSoftmaxBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable result_;
};
struct SparseSparseMatmulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SparseSparseMatmulBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  SavedVariable self_;
};
struct SoftplusBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SoftplusBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar beta;
  SavedVariable self_;
  at::Scalar threshold;
};
struct SoftshrinkBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SoftshrinkBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar lambd;
  SavedVariable self_;
};
struct ThresholdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ThresholdBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  at::Scalar threshold;
};
struct ThresholdBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ThresholdBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  at::Scalar threshold;
};
struct ReflectionPad1DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReflectionPad1DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
};
struct ReflectionPad2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReflectionPad2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
};
struct ReflectionPad3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReflectionPad3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
};
struct ReplicationPad1DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReplicationPad1DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
};
struct ReplicationPad2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReplicationPad2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
};
struct ReplicationPad3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReplicationPad3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
};
struct UpsampleLinear1DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleLinear1DBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct UpsampleBilinear2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleBilinear2DBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct UpsampleBilinear2DAaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleBilinear2DAaBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct UpsampleBicubic2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleBicubic2DBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct UpsampleBicubic2DAaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleBicubic2DAaBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct UpsampleTrilinear3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleTrilinear3DBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct UpsampleNearest1DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleNearest1DBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct UpsampleNearestExact1DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleNearestExact1DBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct UpsampleNearest2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleNearest2DBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct UpsampleNearestExact2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleNearestExact2DBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct UpsampleNearest3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleNearest3DBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct UpsampleNearestExact3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleNearestExact3DBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct PixelShuffleBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PixelShuffleBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t upscale_factor = 0;
};
struct PixelUnshuffleBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PixelUnshuffleBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t downscale_factor = 0;
};
struct AdaptiveAvgPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AdaptiveAvgPool2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct AdaptiveAvgPool3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AdaptiveAvgPool3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct AdaptiveMaxPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AdaptiveMaxPool2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result1_;
};
struct AdaptiveMaxPool3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AdaptiveMaxPool3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable result1_;
};
struct AvgPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AvgPool2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool ceil_mode;
  bool count_include_pad;
  c10::optional<int64_t> divisor_override;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;
};
struct AvgPool3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AvgPool3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool ceil_mode;
  bool count_include_pad;
  c10::optional<int64_t> divisor_override;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;
};
struct FractionalMaxPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FractionalMaxPool2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> output_size;
  SavedVariable self_;
  SavedVariable result1_;
};
struct FractionalMaxPool3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FractionalMaxPool3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> output_size;
  SavedVariable self_;
  SavedVariable result1_;
};
struct LinearBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinearBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable input_;
  SavedVariable weight_;
};
struct LinearBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LinearBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
};
struct MaxPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaxPool2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool ceil_mode;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;
};
struct MpsConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MpsConvolutionBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct MpsConvolutionBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MpsConvolutionBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  SavedVariable grad_output_;
  c10::SymInt groups;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct MaxPool2DWithIndicesBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaxPool2DWithIndicesBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool ceil_mode;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;
  SavedVariable result1_;
};
struct MaxPool3DWithIndicesBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaxPool3DWithIndicesBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool ceil_mode;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;
  SavedVariable result1_;
};
struct MaxUnpool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaxUnpool2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable indices_;
};
struct MaxUnpool3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaxUnpool3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable indices_;
};
struct ConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConvolutionBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  SavedVariable input_;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  std::vector<c10::SymInt> stride;
  bool transposed;
  SavedVariable weight_;
};
struct ConvolutionBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConvolutionBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  SavedVariable input_;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  std::vector<c10::SymInt> stride;
  bool transposed;
  SavedVariable weight_;
};
struct ConvolutionBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConvolutionBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  SavedVariable grad_output_;
  c10::SymInt groups;
  SavedVariable input_;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  std::vector<c10::SymInt> stride;
  bool transposed;
  SavedVariable weight_;
};
struct ConvolutionOverrideableBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConvolutionOverrideableBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  SavedVariable input_;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  std::vector<c10::SymInt> stride;
  bool transposed;
  SavedVariable weight_;
};
struct ConvolutionBackwardOverrideableBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConvolutionBackwardOverrideableBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  SavedVariable grad_output_;
  c10::SymInt groups;
  SavedVariable input_;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  std::vector<c10::SymInt> stride;
  bool transposed;
  SavedVariable weight_;
};
struct SlowConvTranspose2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SlowConvTranspose2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct SlowConvTranspose3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SlowConvTranspose3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct SlowConv2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SlowConv2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> kernel_size;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct SlowConv2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SlowConv2DBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct ConvDepthwise2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConvDepthwise2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct ConvDepthwise3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConvDepthwise3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct SlowConv3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SlowConv3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct SlowConvDilated2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SlowConvDilated2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct SlowConvDilated3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SlowConvDilated3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct Col2ImBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Col2ImBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
};
struct Im2ColBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "Im2ColBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  c10::SymInt self_sym_argsize_minus_1;
  c10::SymInt self_sym_argsize_minus_2;
  std::vector<int64_t> stride;
};
struct AdaptiveAvgPool2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AdaptiveAvgPool2DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::SymInt grad_output_sym_argsize_minus_1;
  c10::SymInt grad_output_sym_argsize_minus_2;
  torch::autograd::generated::TypeAndSize self_info;
};
struct AdaptiveAvgPool3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AdaptiveAvgPool3DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::SymInt grad_output_sym_argsize_minus_1;
  c10::SymInt grad_output_sym_argsize_minus_2;
  c10::SymInt grad_output_sym_argsize_minus_3;
  torch::autograd::generated::TypeAndSize self_info;
};
struct AdaptiveMaxPool2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AdaptiveMaxPool2DBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;
};
struct AdaptiveMaxPool3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AdaptiveMaxPool3DBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;
};
struct AvgPool2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AvgPool2DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool ceil_mode;
  bool count_include_pad;
  c10::optional<int64_t> divisor_override;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;
  std::vector<int64_t> stride;
};
struct AvgPool3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AvgPool3DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool ceil_mode;
  bool count_include_pad;
  c10::optional<int64_t> divisor_override;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  torch::autograd::generated::TypeAndSize self_info;
  std::vector<int64_t> stride;
};
struct EluBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "EluBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_or_result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  SavedVariable grad_output_;
  at::Scalar input_scale;
  bool is_result;
  at::Scalar scale;
  SavedVariable self_or_result_;
};
struct FractionalMaxPool2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FractionalMaxPool2DBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;
};
struct FractionalMaxPool3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FractionalMaxPool3DBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;
};
struct GluBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "GluBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable grad_output_;
  SavedVariable self_;
};
struct HardtanhBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "HardtanhBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar max_val;
  at::Scalar min_val;
  SavedVariable self_;
};
struct LogSigmoidBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LogSigmoidBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.reset_data();
    grad_output_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable buffer_;
  SavedVariable grad_output_;
  SavedVariable self_;
};
struct LogSoftmaxBackwardDataBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LogSoftmaxBackwardDataBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    output_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable grad_output_;
  SavedVariable output_;
};
struct LeakyReluBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LeakyReluBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar negative_slope;
  SavedVariable self_;
};
struct MaxPool2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaxPool2DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  torch::autograd::generated::TypeAndSize self_info;
};
struct MaxPool2DWithIndicesBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaxPool2DWithIndicesBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;
};
struct MaxPool3DWithIndicesBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MaxPool3DWithIndicesBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    indices_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable indices_;
  torch::autograd::generated::TypeAndSize self_info;
};
struct MseLossBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MseLossBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
};
struct NllLossBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NllLossBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    target_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::SymInt ignore_index;
  int64_t reduction = 0;
  SavedVariable target_;
  SavedVariable weight_;
};
struct NllLoss2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NllLoss2DBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    target_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::SymInt ignore_index;
  int64_t reduction = 0;
  SavedVariable target_;
  SavedVariable weight_;
};
struct RreluWithNoiseBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "RreluWithNoiseBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    noise_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar lower;
  SavedVariable noise_;
  SavedVariable self_;
  bool training;
  at::Scalar upper;
};
struct ReflectionPad1DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReflectionPad1DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  torch::autograd::generated::TypeAndSize self_info;
};
struct ReflectionPad2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReflectionPad2DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  torch::autograd::generated::TypeAndSize self_info;
};
struct ReflectionPad3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReflectionPad3DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  torch::autograd::generated::TypeAndSize self_info;
};
struct ReplicationPad1DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReplicationPad1DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  torch::autograd::generated::TypeAndSize self_info;
};
struct ReplicationPad2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReplicationPad2DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  torch::autograd::generated::TypeAndSize self_info;
};
struct ReplicationPad3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReplicationPad3DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> padding;
  torch::autograd::generated::TypeAndSize self_info;
};
struct SparseSampledAddmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SparseSampledAddmmBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mat1_.reset_data();
    mat2_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  at::Scalar beta;
  SavedVariable mat1_;
  SavedVariable mat2_;
  SavedVariable self_;
};
struct SparseMmReduceImplBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SparseMmReduceImplBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  std::string reduce;
  SavedVariable self_;
  SavedVariable result1_;
};
struct SmoothL1LossBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SmoothL1LossBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double beta;
  SavedVariable grad_output_;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
};
struct HuberLossBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "HuberLossBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double delta;
  SavedVariable grad_output_;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
};
struct SoftplusBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SoftplusBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar beta;
  SavedVariable grad_output_;
  SavedVariable self_;
  at::Scalar threshold;
};
struct SoftmaxBackwardDataBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SoftmaxBackwardDataBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    output_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable grad_output_;
  at::ScalarType input_dtype;
  SavedVariable output_;
};
struct SoftMarginLossBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SoftMarginLossBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    self_.reset_data();
    target_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  int64_t reduction = 0;
  SavedVariable self_;
  SavedVariable target_;
};
struct SoftshrinkBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SoftshrinkBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar lambd;
  SavedVariable self_;
};
struct ThresholdBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ThresholdBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  at::Scalar threshold;
};
struct UpsampleLinear1DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleLinear1DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales;
};
struct UpsampleBilinear2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleBilinear2DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
};
struct UpsampleBilinear2DAaBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleBilinear2DAaBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
};
struct UpsampleBicubic2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleBicubic2DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
};
struct UpsampleBicubic2DAaBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleBicubic2DAaBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
};
struct UpsampleTrilinear3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleTrilinear3DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool align_corners;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
};
struct UpsampleNearest1DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleNearest1DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales;
};
struct UpsampleNearestExact1DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleNearestExact1DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales;
};
struct UpsampleNearest2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleNearest2DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
};
struct UpsampleNearestExact2DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleNearestExact2DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
};
struct UpsampleNearest3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleNearest3DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
};
struct UpsampleNearestExact3DBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UpsampleNearestExact3DBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> output_size;
  c10::optional<double> scales_d;
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
};
struct SigmoidBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SigmoidBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    output_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  SavedVariable output_;
};
struct TanhBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TanhBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    output_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable grad_output_;
  SavedVariable output_;
};
struct CudnnCtcLossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CudnnCtcLossBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result0_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;
};
struct CudnnCtcLossBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CudnnCtcLossBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result0_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;
};
struct CudnnConvolutionTransposeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CudnnConvolutionTransposeBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct MpsConvolutionTransposeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MpsConvolutionTransposeBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct CudnnConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CudnnConvolutionBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct CudnnGridSamplerBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CudnnGridSamplerBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grid_.reset_data();
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable grid_;
  SavedVariable self_;
};
struct CudnnAffineGridGeneratorBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CudnnAffineGridGeneratorBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t C = 0;
  int64_t H = 0;
  int64_t N = 0;
  int64_t W = 0;
};
struct CudnnBatchNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CudnnBatchNormBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double epsilon;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;
};
struct CudnnBatchNormBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CudnnBatchNormBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    input_.reset_data();
    reserveSpace_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    save_mean_.reset_data();
    save_var_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double epsilon;
  SavedVariable grad_output_;
  SavedVariable input_;
  SavedVariable reserveSpace_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_var_;
  SavedVariable weight_;
};
struct NnpackSpatialConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NnpackSpatialConvolutionBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  SavedVariable input_;
  std::vector<c10::SymInt> padding;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct LstmMpsBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LstmMpsBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    hx_.clear();
    hx_released_ = true;
    input_.reset_data();
    params_.clear();
    params_released_ = true;
    result3_.reset_data();
    result4_.reset_data();
    result5_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool batch_first;
  bool bidirectional;
  double dropout;
  bool has_biases;
  std::vector<SavedVariable> hx_;
  bool hx_released_ = false;
  SavedVariable input_;
  int64_t num_layers = 0;
  std::vector<SavedVariable> params_;
  bool params_released_ = false;
  bool train;
  SavedVariable result3_;
  SavedVariable result4_;
  SavedVariable result5_;
  size_t hx_size_;
  size_t params_size_;
};
struct CudnnRnnBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CudnnRnnBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    cx_.reset_data();
    dropout_state_.reset_data();
    hx_.reset_data();
    input_.reset_data();
    weight_.clear();
    weight_released_ = true;
    result0_.reset_data();
    result3_.reset_data();
    result4_.reset_data();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool batch_first;
  std::vector<c10::SymInt> batch_sizes;
  bool bidirectional;
  SavedVariable cx_;
  double dropout;
  SavedVariable dropout_state_;
  c10::SymInt hidden_size;
  SavedVariable hx_;
  SavedVariable input_;
  int64_t mode = 0;
  int64_t num_layers = 0;
  c10::SymInt proj_size;
  bool train;
  std::vector<SavedVariable> weight_;
  bool weight_released_ = false;
  int64_t weight_stride0 = 0;
  SavedVariable result0_;
  SavedVariable result3_;
  SavedVariable result4_;
  size_t weight_size_;
};
struct CudnnRnnBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "CudnnRnnBackwardBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  size_t weight_size_;
};
struct MiopenConvolutionTransposeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MiopenConvolutionTransposeBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> output_padding;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct MiopenConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MiopenConvolutionBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct MiopenDepthwiseConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MiopenDepthwiseConvolutionBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct MiopenBatchNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MiopenBatchNormBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    weight_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double epsilon;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  SavedVariable weight_;
  SavedVariable result1_;
  SavedVariable result2_;
};
struct MiopenBatchNormBackwardBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MiopenBatchNormBackwardBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_output_.reset_data();
    input_.reset_data();
    running_mean_.reset_data();
    running_var_.reset_data();
    save_mean_.reset_data();
    save_var_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double epsilon;
  SavedVariable grad_output_;
  SavedVariable input_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_var_;
  SavedVariable weight_;
};
struct MiopenRnnBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MiopenRnnBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    cx_.reset_data();
    dropout_state_.reset_data();
    hx_.reset_data();
    input_.reset_data();
    weight_.clear();
    weight_released_ = true;
    result0_.reset_data();
    result3_.reset_data();
    result4_.reset_data();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool batch_first;
  std::vector<int64_t> batch_sizes;
  bool bidirectional;
  SavedVariable cx_;
  double dropout;
  SavedVariable dropout_state_;
  int64_t hidden_size = 0;
  SavedVariable hx_;
  SavedVariable input_;
  int64_t mode = 0;
  int64_t num_layers = 0;
  bool train;
  std::vector<SavedVariable> weight_;
  bool weight_released_ = false;
  int64_t weight_stride0 = 0;
  SavedVariable result0_;
  SavedVariable result3_;
  SavedVariable result4_;
  size_t weight_size_;
};
struct MkldnnRnnLayerBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MkldnnRnnLayerBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    cx__.reset_data();
    hx__.reset_data();
    input_.reset_data();
    weight0_.reset_data();
    weight1_.reset_data();
    weight2_.reset_data();
    weight3_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool batch_first;
  std::vector<int64_t> batch_sizes;
  bool bidirectional;
  SavedVariable cx__;
  bool has_biases;
  int64_t hidden_size = 0;
  SavedVariable hx__;
  SavedVariable input_;
  int64_t mode = 0;
  int64_t num_layers = 0;
  bool reverse;
  bool train;
  SavedVariable weight0_;
  SavedVariable weight1_;
  SavedVariable weight2_;
  SavedVariable weight3_;
  SavedVariable result0_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;
};
struct MkldnnConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MkldnnConvolutionBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  c10::OptionalArray<c10::SymInt> bias_sym_sizes_opt;
  std::vector<c10::SymInt> dilation;
  c10::SymInt groups;
  std::vector<c10::SymInt> padding;
  SavedVariable self_;
  std::vector<c10::SymInt> stride;
  SavedVariable weight_;
};
struct MkldnnLinearBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MkldnnLinearBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    weight_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  SavedVariable weight_;
};
struct MkldnnMaxPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MkldnnMaxPool2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool ceil_mode;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;
  SavedVariable result_;
};
struct MkldnnMaxPool3DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MkldnnMaxPool3DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool ceil_mode;
  std::vector<int64_t> dilation;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> padding;
  SavedVariable self_;
  std::vector<int64_t> stride;
  SavedVariable result_;
};
struct MkldnnAdaptiveAvgPool2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MkldnnAdaptiveAvgPool2DBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct MkldnnReshapeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "MkldnnReshapeBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct NestedTensorFromTensorListBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NestedTensorFromTensorListBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    list_.clear();
    list_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> list_;
  bool list_released_ = false;
  size_t list_size_;
};
struct NestedTensorFromMaskBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NestedTensorFromMaskBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> t_sym_sizes;
};
struct NestedFromPaddedBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NestedFromPaddedBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    padded_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool fuse_transform_0213;
  SavedVariable padded_;
};
struct ToPaddedTensorBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ToPaddedTensorBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct NestedViewFromBufferBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NestedViewFromBufferBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct ScaledDotProductEfficientAttentionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ScaledDotProductEfficientAttentionBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    attn_bias_.reset_data();
    key_.reset_data();
    query_.reset_data();
    value_.reset_data();
    log_sumexp_.reset_data();
    output_.reset_data();
    philox_offset_.reset_data();
    philox_seed_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable attn_bias_;
  double dropout_p;
  bool is_causal;
  SavedVariable key_;
  SavedVariable query_;
  c10::optional<double> scale;
  SavedVariable value_;
  SavedVariable log_sumexp_;
  SavedVariable output_;
  SavedVariable philox_offset_;
  SavedVariable philox_seed_;
};
struct ScaledDotProductFlashAttentionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ScaledDotProductFlashAttentionBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    key_.reset_data();
    query_.reset_data();
    value_.reset_data();
    cum_seq_k_.reset_data();
    cum_seq_q_.reset_data();
    logsumexp_.reset_data();
    output_.reset_data();
    philox_offset_.reset_data();
    philox_seed_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  double dropout_p;
  bool is_causal;
  SavedVariable key_;
  SavedVariable query_;
  c10::optional<double> scale;
  SavedVariable value_;
  SavedVariable cum_seq_k_;
  SavedVariable cum_seq_q_;
  SavedVariable logsumexp_;
  c10::SymInt max_k;
  c10::SymInt max_q;
  SavedVariable output_;
  SavedVariable philox_offset_;
  SavedVariable philox_seed_;
};
struct FlashAttentionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FlashAttentionBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    cum_seq_k_.reset_data();
    cum_seq_q_.reset_data();
    key_.reset_data();
    query_.reset_data();
    value_.reset_data();
    output_.reset_data();
    philox_offset_.reset_data();
    philox_seed_.reset_data();
    softmax_logsumexp_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable cum_seq_k_;
  SavedVariable cum_seq_q_;
  double dropout_p;
  bool is_causal;
  SavedVariable key_;
  c10::SymInt max_k;
  c10::SymInt max_q;
  SavedVariable query_;
  c10::optional<double> scale;
  SavedVariable value_;
  SavedVariable output_;
  SavedVariable philox_offset_;
  SavedVariable philox_seed_;
  SavedVariable softmax_logsumexp_;
};
struct EfficientAttentionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "EfficientAttentionBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    bias_.reset_data();
    cu_seqlens_k_.reset_data();
    cu_seqlens_q_.reset_data();
    key_.reset_data();
    query_.reset_data();
    value_.reset_data();
    logsumexp_.reset_data();
    output_.reset_data();
    philox_offset_.reset_data();
    philox_seed_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable bias_;
  SavedVariable cu_seqlens_k_;
  SavedVariable cu_seqlens_q_;
  int64_t custom_mask_type = 0;
  double dropout_p;
  SavedVariable key_;
  SavedVariable query_;
  c10::optional<double> scale;
  SavedVariable value_;
  SavedVariable logsumexp_;
  c10::SymInt max_seqlen_batch_k;
  c10::SymInt max_seqlen_batch_q;
  SavedVariable output_;
  SavedVariable philox_offset_;
  SavedVariable philox_seed_;
};
struct FftR2CBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FftR2CBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  int64_t normalization = 0;
  bool onesided;
  SavedVariable self_;
};
struct FftC2RBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FftC2RBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  int64_t normalization = 0;
};
struct FftC2CBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "FftC2CBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> dim;
  bool forward;
  int64_t normalization = 0;
};
struct UnbindBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnbindBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
};
struct UnbindBackwardAutogradNestedTensor0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnbindBackwardAutogradNestedTensor0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  at::TensorOptions self_options;
};
struct StackBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "StackBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  ::std::vector<at::ScalarType> tensors_args_scalartypes;
  size_t tensors_size_;
};
struct ThnnFusedLstmCellBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ThnnFusedLstmCellBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    cx_.reset_data();
    hidden_bias_.reset_data();
    hidden_gates_.reset_data();
    input_bias_.reset_data();
    input_gates_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable cx_;
  SavedVariable hidden_bias_;
  SavedVariable hidden_gates_;
  SavedVariable input_bias_;
  SavedVariable input_gates_;
  SavedVariable result1_;
  SavedVariable result2_;
};
struct ThnnFusedGruCellBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ThnnFusedGruCellBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    hidden_bias_.reset_data();
    hidden_gates_.reset_data();
    hx_.reset_data();
    input_bias_.reset_data();
    input_gates_.reset_data();
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable hidden_bias_;
  SavedVariable hidden_gates_;
  SavedVariable hx_;
  SavedVariable input_bias_;
  SavedVariable input_gates_;
  SavedVariable result1_;
};
struct PackPaddedSequenceBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PackPaddedSequenceBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  bool batch_first;
  std::vector<c10::SymInt> input_sym_sizes;
  SavedVariable result1_;
};
struct SegmentReduceBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SegmentReduceBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    data_.reset_data();
    lengths_.reset_data();
    offsets_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t axis = 0;
  SavedVariable data_;
  c10::optional<at::Scalar> initial;
  SavedVariable lengths_;
  SavedVariable offsets_;
  std::string reduce;
  SavedVariable result_;
};
struct PinMemoryBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PinMemoryBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct TestWarnInAutogradBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TestWarnInAutogradBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct TestAutogradMultipleDispatchBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TestAutogradMultipleDispatchBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct TestAutogradMultipleDispatchBackwardAutogradNestedTensor0
    : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TestAutogradMultipleDispatchBackwardAutogradNestedTensor0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct TestAutogradMultipleDispatchBackwardAutogradCUDA0
    : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TestAutogradMultipleDispatchBackwardAutogradCUDA0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct TestAutogradMultipleDispatchBackwardAutogradNestedTensor1
    : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TestAutogradMultipleDispatchBackwardAutogradNestedTensor1";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct TestAutogradMultipleDispatchViewBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TestAutogradMultipleDispatchViewBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct TestAutogradMultipleDispatchViewBackwardAutogradCUDA0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TestAutogradMultipleDispatchViewBackwardAutogradCUDA0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct ScatterReduceBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ScatterReduceBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.reset_data();
    self_.reset_data();
    src_.reset_data();
    result_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  bool include_self;
  SavedVariable index_;
  std::string reduce;
  SavedVariable self_;
  SavedVariable src_;
  SavedVariable result_;
};
struct ReshapeCopyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReshapeCopyBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct ForeachDivBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachDivBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
struct ForeachPowBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachPowBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.clear();
    exponent_released_ = true;
    self_.clear();
    self_released_ = true;
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> exponent_;
  bool exponent_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
  size_t exponent_size_;
};
struct ForeachPowBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachPowBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<at::Scalar> exponent;
  bool exponent_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachPowBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachPowBackward2";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    exponent_.clear();
    exponent_released_ = true;
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> exponent_;
  bool exponent_released_ = false;
  at::Scalar self;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t exponent_size_;
};
struct ForeachMinimumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachMinimumBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar scalar;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachMinimumBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachMinimumBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachMaximumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachMaximumBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar scalar;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachMaximumBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachMaximumBackward1";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachNormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachNormBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar ord;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
struct AliasBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AliasBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct AsStridedBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "AsStridedBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::TensorGeometry self_geometry;
  std::vector<c10::SymInt> size;
  c10::optional<c10::SymInt> storage_offset;
  std::vector<c10::SymInt> stride;
};
struct ConjBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ConjBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct NegViewBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NegViewBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct DiagonalBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "DiagonalBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim1 = 0;
  int64_t dim2 = 0;
  int64_t offset = 0;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct ExpandBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ExpandBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct PermuteBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "PermuteBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dims;
};
struct ReshapeAliasBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ReshapeAliasBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SelectBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SelectBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt index;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SelectBackwardAutogradNestedTensor0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SelectBackwardAutogradNestedTensor0_copy";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::SymInt index;
  SavedVariable self_;
};
struct SliceBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SliceBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  c10::optional<c10::SymInt> end;
  std::vector<c10::SymInt> self_sym_sizes;
  c10::optional<c10::SymInt> start;
  c10::SymInt step;
};
struct SplitBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SplitBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
  c10::SymInt split_size;
};
struct SplitWithSizesBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SplitWithSizesBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> self_sym_sizes;
  std::vector<c10::SymInt> split_sizes;
};
struct SplitWithSizesBackwardAutogradNestedTensor0_copy
    : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SplitWithSizesBackwardAutogradNestedTensor0_copy";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  at::TensorOptions self_options;
  std::vector<c10::SymInt> split_sizes;
};
struct SqueezeBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SqueezeBackward1_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackward1_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SqueezeBackwardAutogradNestedTensor0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackwardAutogradNestedTensor0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
};
struct SqueezeBackward2_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackward2_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct SqueezeBackwardAutogradNestedTensor1_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "SqueezeBackwardAutogradNestedTensor1_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<int64_t> dim;
  int64_t self_dim = 0;
};
struct TBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct TransposeBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TransposeBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim0 = 0;
  int64_t dim1 = 0;
};
struct UnfoldBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnfoldBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dimension = 0;
  std::vector<c10::SymInt> self_sym_sizes;
  int64_t size = 0;
  int64_t step = 0;
};
struct LiftFreshBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "LiftFreshBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct UnsqueezeBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnsqueezeBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
};
struct ViewBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ViewBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct ViewBackwardAutogradNestedTensor0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ViewBackwardAutogradNestedTensor0_copy";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct ViewAsRealBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ViewAsRealBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct ViewAsComplexBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ViewAsComplexBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct ValuesBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ValuesBackward0_copy";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
  std::vector<c10::SymInt> self_sym_sizes;
};
struct ValuesBackwardAutogradNestedTensor0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ValuesBackwardAutogradNestedTensor0_copy";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct NestedViewFromBufferBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "NestedViewFromBufferBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
};
struct UnbindBackward0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnbindBackward0_copy";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
};
struct UnbindBackwardAutogradNestedTensor0_copy : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "UnbindBackwardAutogradNestedTensor0_copy";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  int64_t dim = 0;
  SavedVariable self_;
  at::TensorOptions self_options;
};
struct TestAutogradMultipleDispatchViewBackward0_copy
    : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TestAutogradMultipleDispatchViewBackward0_copy";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy
    : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable self_;
};
struct ForeachAbsBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachAbsBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachAcosBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachAcosBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachAddBackward1Scalar : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachAddBackward1Scalar";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachAddBackward0List : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachAddBackward0List";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
struct ForeachAddBackward1ScalarList : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachAddBackward1ScalarList";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachAddBackward0Tensor : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachAddBackward0Tensor";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  SavedVariable other_;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachAddcdivBackward0Scalar : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachAddcdivBackward0Scalar";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
    tensor1_.clear();
    tensor1_released_ = true;
    tensor2_.clear();
    tensor2_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> tensor1_;
  bool tensor1_released_ = false;
  std::vector<SavedVariable> tensor2_;
  bool tensor2_released_ = false;
  at::Scalar value;
  size_t self_size_;
  size_t tensor1_size_;
  size_t tensor2_size_;
};
struct ForeachAddcdivBackward0ScalarList : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachAddcdivBackward0ScalarList";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
    tensor1_.clear();
    tensor1_released_ = true;
    tensor2_.clear();
    tensor2_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> tensor1_;
  bool tensor1_released_ = false;
  std::vector<SavedVariable> tensor2_;
  bool tensor2_released_ = false;
  size_t self_size_;
  size_t tensor1_size_;
  size_t tensor2_size_;
};
struct ForeachAddcmulBackward0Scalar : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachAddcmulBackward0Scalar";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
    tensor1_.clear();
    tensor1_released_ = true;
    tensor2_.clear();
    tensor2_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> tensor1_;
  bool tensor1_released_ = false;
  std::vector<SavedVariable> tensor2_;
  bool tensor2_released_ = false;
  at::Scalar value;
  size_t self_size_;
  size_t tensor1_size_;
  size_t tensor2_size_;
};
struct ForeachAddcmulBackward0ScalarList : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachAddcmulBackward0ScalarList";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
    tensor1_.clear();
    tensor1_released_ = true;
    tensor2_.clear();
    tensor2_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> tensor1_;
  bool tensor1_released_ = false;
  std::vector<SavedVariable> tensor2_;
  bool tensor2_released_ = false;
  size_t self_size_;
  size_t tensor1_size_;
  size_t tensor2_size_;
};
struct ForeachAsinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachAsinBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachAtanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachAtanBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachCeilBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachCeilBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  size_t self_size_;
};
struct ForeachClampMaxBackward0Scalar : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachClampMaxBackward0Scalar";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar scalar;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachClampMaxBackward1List : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachClampMaxBackward1List";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
struct ForeachClampMaxBackward0ScalarList : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachClampMaxBackward0ScalarList";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachClampMinBackward0Scalar : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachClampMinBackward0Scalar";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar scalar;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachClampMinBackward1List : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachClampMinBackward1List";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
struct ForeachClampMinBackward0ScalarList : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachClampMinBackward0ScalarList";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachCosBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachCosBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachCoshBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachCoshBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachDivBackward1Scalar : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachDivBackward1Scalar";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar scalar;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachDivBackward1ScalarList : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachDivBackward1ScalarList";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachDivBackward0Tensor : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachDivBackward0Tensor";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachErfBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachErfBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachErfcBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachErfcBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachExpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachExpBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
struct ForeachExpm1Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachExpm1Backward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
struct ForeachFloorBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachFloorBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  size_t self_size_;
};
struct ForeachFracBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachFracBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  size_t self_size_;
};
struct ForeachLerpBackward1List : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachLerpBackward1List";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
    tensors1_.clear();
    tensors1_released_ = true;
    weights_.clear();
    weights_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  std::vector<SavedVariable> tensors1_;
  bool tensors1_released_ = false;
  std::vector<SavedVariable> weights_;
  bool weights_released_ = false;
  size_t self_size_;
  size_t tensors1_size_;
  size_t weights_size_;
};
struct ForeachLerpBackward0Scalar : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachLerpBackward0Scalar";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar weight;
  size_t self_size_;
  size_t tensors1_size_;
};
struct ForeachLgammaBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachLgammaBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachLogBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachLogBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachLog10Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachLog10Backward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachLog1PBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachLog1PBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachLog2Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachLog2Backward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachMaximumBackward0List : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachMaximumBackward0List";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
struct ForeachMinimumBackward0List : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachMinimumBackward0List";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
struct ForeachMulBackward1Scalar : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachMulBackward1Scalar";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar scalar;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachMulBackward0List : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachMulBackward0List";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
struct ForeachMulBackward1ScalarList : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachMulBackward1ScalarList";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    scalars.clear();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<at::Scalar> scalars;
  bool scalars_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachMulBackward0Tensor : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachMulBackward0Tensor";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  SavedVariable other_;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachNegBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachNegBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  size_t self_size_;
};
struct ForeachPowBackward0Scalar : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachPowBackward0Scalar";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar exponent;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachReciprocalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachReciprocalBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
struct ForeachRoundBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachRoundBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  size_t self_size_;
};
struct ForeachSigmoidBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachSigmoidBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
struct ForeachSignBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachSignBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  size_t self_size_;
};
struct ForeachSinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachSinBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachSinhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachSinhBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachSqrtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachSqrtBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
struct ForeachSubBackward1Scalar : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachSubBackward1Scalar";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachSubBackward0List : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachSubBackward0List";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.clear();
    other_released_ = true;
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  at::Scalar alpha;
  std::vector<SavedVariable> other_;
  bool other_released_ = false;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
  size_t other_size_;
};
struct ForeachSubBackward1ScalarList : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachSubBackward1ScalarList";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.clear();
    self_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> self_;
  bool self_released_ = false;
  size_t self_size_;
};
struct ForeachTanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachTanBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
struct ForeachTanhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachTanhBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.clear();
    result_released_ = true;
  }

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;
  std::vector<SavedVariable> result_;
  bool result_released_ = false;
  size_t self_size_;
};
struct ForeachTruncBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ForeachTruncBackward0";
  }
  void release_variables() override {}

  void compiled_args(CompiledNodeArgs& args) override;
  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override;

  size_t self_size_;
};

} // namespace generated
} // namespace autograd
} // namespace torch
