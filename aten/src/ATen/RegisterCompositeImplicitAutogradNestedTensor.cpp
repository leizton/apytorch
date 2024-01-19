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
#include <ATen/CompositeImplicitAutogradNestedTensorFunctions.h>

// See template file RegisterDispatchDefinitions.ini
namespace at {
// NB: TORCH_LIBRARY_IMPL must be in an anonymous namespace to avoid
// ambiguity with conflicting identifiers that may have been defined in
// at namespace already.
namespace {
namespace {
at::Tensor wrapper_CompositeImplicitAutogradNestedTensor__randn_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    // No device check
  // DeviceGuard omitted
  return at::native::randn_like(self, dtype, layout, device, pin_memory, memory_format);
}
} // anonymous namespace
namespace {
at::Tensor wrapper_CompositeImplicitAutogradNestedTensor__reshape(const at::Tensor & self, c10::SymIntArrayRef shape) {
    // No device check
  // DeviceGuard omitted
  return at::native::reshape_nested_symint(self, shape);
}
} // anonymous namespace
namespace {
at::Tensor wrapper_CompositeImplicitAutogradNestedTensor__reshape_as(const at::Tensor & self, const at::Tensor & other) {
    // No device check
  // DeviceGuard omitted
  return at::native::reshape_as_nested(self, other);
}
} // anonymous namespace
namespace {
at::Tensor wrapper_CompositeImplicitAutogradNestedTensor__zeros_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    // No device check
  // DeviceGuard omitted
  return at::native::zeros_like(self, dtype, layout, device, pin_memory, memory_format);
}
} // anonymous namespace
TORCH_LIBRARY_IMPL(aten, CompositeImplicitAutogradNestedTensor, m) {
    m.impl("randn_like",
TORCH_FN(wrapper_CompositeImplicitAutogradNestedTensor__randn_like));
m.impl("reshape",
TORCH_FN(wrapper_CompositeImplicitAutogradNestedTensor__reshape));
m.impl("reshape_as",
TORCH_FN(wrapper_CompositeImplicitAutogradNestedTensor__reshape_as));
m.impl("zeros_like",
TORCH_FN(wrapper_CompositeImplicitAutogradNestedTensor__zeros_like));
};
} // anonymous namespace
namespace compositeimplicitautogradnestedtensor {
at::Tensor randn_like(const at::Tensor & self, at::TensorOptions options, c10::optional<at::MemoryFormat> memory_format) {
return wrapper_CompositeImplicitAutogradNestedTensor__randn_like(self, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}
at::Tensor randn_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
return wrapper_CompositeImplicitAutogradNestedTensor__randn_like(self, dtype, layout, device, pin_memory, memory_format);
}
at::Tensor reshape(const at::Tensor & self, at::IntArrayRef shape) {
return wrapper_CompositeImplicitAutogradNestedTensor__reshape(self, c10::fromIntArrayRefSlow(shape));
}
at::Tensor reshape_symint(const at::Tensor & self, c10::SymIntArrayRef shape) {
return wrapper_CompositeImplicitAutogradNestedTensor__reshape(self, shape);
}
at::Tensor reshape_as(const at::Tensor & self, const at::Tensor & other) {
return wrapper_CompositeImplicitAutogradNestedTensor__reshape_as(self, other);
}
at::Tensor zeros_like(const at::Tensor & self, at::TensorOptions options, c10::optional<at::MemoryFormat> memory_format) {
return wrapper_CompositeImplicitAutogradNestedTensor__zeros_like(self, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}
at::Tensor zeros_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
return wrapper_CompositeImplicitAutogradNestedTensor__zeros_like(self, dtype, layout, device, pin_memory, memory_format);
}
} // namespace compositeimplicitautogradnestedtensor
} // namespace at