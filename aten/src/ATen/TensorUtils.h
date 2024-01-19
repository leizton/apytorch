#pragma once

#include <ATen/DimVector.h>
#include <ATen/EmptyTensor.h>
#include <ATen/Tensor.h>
#include <ATen/TensorGeometry.h>
#include <ATen/Utils.h>

#include <utility>

// These functions are NOT in Utils.h, because this file has a dep on Tensor.h

#define TORCH_CHECK_TENSOR_ALL(cond, ...) \
  TORCH_CHECK((cond)._is_all_true().item<bool>(), __VA_ARGS__);

namespace at {

// The following are utility functions for checking that arguments
// make sense.  These are particularly useful for native functions,
// which do NO argument checking by default.

struct TensorArg {
  const Tensor& tensor;
  const char* name;
  int pos; // 1-indexed
  TensorArg(const Tensor& tensor, const char* name, int pos)
      : tensor(tensor), name(name), pos(pos) {}
  // Try to mitigate any possibility of dangling reference to temporaries.
  TensorArg(Tensor&& tensor, const char* name, int pos) = delete;
  const Tensor* operator->() const {
    return &tensor;
  }
  const Tensor& operator*() const {
    return tensor;
  }
};

struct TensorGeometryArg {
  TensorGeometry tensor;
  const char* name;
  int pos; // 1-indexed
  /* implicit */ TensorGeometryArg(TensorArg arg)
      : tensor(TensorGeometry{arg.tensor}), name(arg.name), pos(arg.pos) {}
  TensorGeometryArg(TensorGeometry tensor, const char* name, int pos)
      : tensor(std::move(tensor)), name(name), pos(pos) {}
  const TensorGeometry* operator->() const {
    return &tensor;
  }
  const TensorGeometry& operator*() const {
    return tensor;
  }
};

// A string describing which function did checks on its input
// arguments.
// TODO: Consider generalizing this into a call stack.
using CheckedFrom = const char*;

// The undefined convention: singular operators assume their arguments
// are defined, but functions which take multiple tensors will
// implicitly filter out undefined tensors (to make it easier to perform
// tests which should apply if the tensor is defined, and should not
// otherwise.)
//
// NB: This means that the n-ary operators take lists of TensorArg,
// not TensorGeometryArg, because the Tensor to TensorGeometry
// conversion will blow up if you have undefined tensors.

std::ostream& operator<<(std::ostream& out, TensorGeometryArg t);
void checkDim(
    CheckedFrom c,
    const Tensor& tensor,
    const char* name,
    int pos, // 1-indexed
    int64_t dim);
void checkDim(CheckedFrom c, const TensorGeometryArg& t, int64_t dim);
// NB: this is an inclusive-exclusive range
void checkDimRange(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim_start,
    int64_t dim_end);
void checkSameDim(
    CheckedFrom c,
    const TensorGeometryArg& t1,
    const TensorGeometryArg& t2);
void checkContiguous(CheckedFrom c, const TensorGeometryArg& t);
void checkAllContiguous(CheckedFrom c, at::ArrayRef<TensorArg> ts);
void checkSize(
    CheckedFrom c,
    const TensorGeometryArg& t,
    IntArrayRef sizes);
void checkSize_symint(
    CheckedFrom c,
    const TensorGeometryArg& t,
    c10::SymIntArrayRef sizes);
void checkSize(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim,
    int64_t size);
void checkSize_symint(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim,
    c10::SymInt size);
void checkNumel(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t numel);
void checkSameNumel(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
void checkAllSameNumel(CheckedFrom c, ArrayRef<TensorArg> tensors);
void checkScalarType(CheckedFrom c, const TensorArg& t, ScalarType s);
void checkScalarTypes(
    CheckedFrom c,
    const TensorArg& t,
    at::ArrayRef<ScalarType> l);
void checkSameGPU(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
void checkAllSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors);
void checkSameType(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
void checkAllSameType(CheckedFrom c, ArrayRef<TensorArg> tensors);
void checkSameSize(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
void checkAllSameSize(CheckedFrom c, ArrayRef<TensorArg> tensors);
void checkDefined(CheckedFrom c, const TensorArg& t);
void checkAllDefined(CheckedFrom c, at::ArrayRef<TensorArg> t);

// FixMe: does TensorArg slow things down?
void checkBackend(
    CheckedFrom c,
    at::ArrayRef<Tensor> t,
    at::Backend backend);

void checkDeviceType(
    CheckedFrom c,
    at::ArrayRef<Tensor> tensors,
    at::DeviceType device_type);

void checkLayout(CheckedFrom c, const Tensor& t, Layout layout);

void checkLayout(
    CheckedFrom c,
    at::ArrayRef<Tensor> tensors,
    at::Layout layout);

// Methods for getting data_ptr if tensor is defined
void* maybe_data_ptr(const Tensor& tensor);
void* maybe_data_ptr(const TensorArg& tensor);

void check_dim_size(
    const Tensor& tensor,
    int64_t dim,
    int64_t dim_size,
    int64_t size);

namespace detail {
std::vector<int64_t> defaultStrides(IntArrayRef sizes);

c10::optional<std::vector<int64_t>> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    IntArrayRef newshape);

c10::optional<SymDimVector> computeStride(
    c10::SymIntArrayRef oldshape,
    c10::SymIntArrayRef oldstride,
    c10::SymIntArrayRef newshape);

c10::optional<DimVector> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    const DimVector& newshape);

} // namespace detail
} // namespace at
