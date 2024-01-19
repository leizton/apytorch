#pragma once

#include <c10/util/Exception.h>

#include <c10/util/Registry.h>

#include <string>

namespace at {
class Context;
}

namespace at {

constexpr const char* MTIA_HELP =
    "The MTIA backend requires MTIA extension for PyTorch;"
    "this error has occurred because you are trying "
    "to use some MTIA's functionality without MTIA extension included.";

struct MTIAHooksInterface {
  virtual ~MTIAHooksInterface() = default;

  virtual void initMTIA() const {
    TORCH_CHECK(
        false,
        "Cannot initialize MTIA without MTIA Extension for PyTorch.",
        MTIA_HELP);
  }

  virtual bool hasMTIA() const {
    return false;
  }

  virtual std::string showConfig() const {
    TORCH_CHECK(
        false,
        "Cannot query detailed MTIA version without MTIA Extension for PyTorch.",
        MTIA_HELP);
  }
};

struct MTIAHooksArgs {};

C10_DECLARE_REGISTRY(MTIAHooksRegistry, MTIAHooksInterface, MTIAHooksArgs);
#define REGISTER_MTIA_HOOKS(clsname) \
  C10_REGISTER_CLASS(MTIAHooksRegistry, clsname, clsname)

namespace detail {
const MTIAHooksInterface& getMTIAHooks();
} // namespace detail
} // namespace at
