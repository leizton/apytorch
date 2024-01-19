// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

// an external backend might generate file within its code tree
// and check all the source files within the tree with clang-format.
// so, disable it since the backend might have a different config.
// clang-format off

// NOTE: This condition is true for all PyTorch internal libraries, it
//       just excludes external projects such as torch_xla which
//       re-use some of the PyTorch codegen machinery.
#if defined(CAFFE2_BUILD_MAIN_LIB)        || \
    defined(TORCH_CUDA_BUILD_MAIN_LIB)    || \
    defined(TORCH_HIP_BUILD_MAIN_LIB)     || \
    defined(TORCH_CUDA_CU_BUILD_MAIN_LIB) || \
    defined(TORCH_CUDA_CPP_BUILD_MAIN_LIB)
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#endif

// @generated by torchgen/gen.py from RegisterDispatchKey.cpp

#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/Dispatch.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Half.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/Tensor.h>
#include <ATen/native/Resize.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>


#include <ATen/NativeFunctions.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

// See template file RegisterDispatchDefinitions.ini
namespace at {
// NB: TORCH_LIBRARY_IMPL must be in an anonymous namespace to avoid
// ambiguity with conflicting identifiers that may have been defined in
// at namespace already.
namespace {
void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  TORCH_CHECK(options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");
  TORCH_CHECK(options.device() == out.device(),
      "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: avoid the redispatch here
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }
  }
}
void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {
  // These checks are needed on those operators that:
  //   1) don't use 'TensorIterator' (e.g. 'addmm' and 'baddbmm')
  //   2) have particular typing rules (e.g. 'cumsum' and 'cumprod')
  // For other operators (e.g. 'add'), 'TensorIterator' already checks
  // these things separately.
  TORCH_CHECK(options.dtype() == self.dtype(),
      "Bad in-place call: ",
      "input tensor dtype ", self.dtype(), " and output tensor dtype ", options.dtype(), " should match");
  TORCH_CHECK(options.device() == self.device(),
      "Bad in-place call: ",
      "input tensor device ", self.device(), " and output tensor device ", options.device(), " should match");
  TORCH_CHECK(sizes == self.sizes(),
      "Bad in-place call: ",
      "input tensor size ", self.sizes(), " and output tensor size ", sizes, " should match");
}
namespace {
at::Tensor wrapper_NestedTensorMeta___nested_tensor_storage_offsets(const at::Tensor & self) {
    // No device check
  // DeviceGuard omitted
  return at::native::_nested_tensor_storage_offsets(self);
}
} // anonymous namespace
TORCH_LIBRARY_IMPL(aten, NestedTensorMeta, m) {
    m.impl("_nested_tensor_storage_offsets",
TORCH_FN(wrapper_NestedTensorMeta___nested_tensor_storage_offsets));
};
} // anonymous namespace
namespace nestedtensormeta {
at::Tensor _nested_tensor_storage_offsets(const at::Tensor & self) {
return wrapper_NestedTensorMeta___nested_tensor_storage_offsets(self);
}
} // namespace nestedtensormeta
} // namespace at
