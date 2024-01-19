#pragma once

#include <c10/core/Allocator.h>
#include <ATen/core/Generator.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
namespace at {

struct PrivateUse1HooksInterface {
  virtual ~PrivateUse1HooksInterface() = default;
  virtual const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getDefaultGenerator`.");
  }

  virtual at::Device getDeviceFromPtr(void* data) const {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getDeviceFromPtr`.");
  }

  virtual Allocator* getPinnedMemoryAllocator() const {
    TORCH_CHECK(false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getPinnedMemoryAllocator`.");
  }

  virtual void initPrivateUse1() const {}
};

struct PrivateUse1HooksArgs {};

void RegisterPrivateUse1HooksInterface(at::PrivateUse1HooksInterface* hook_);

at::PrivateUse1HooksInterface* GetPrivateUse1HooksInterface();

bool isPrivateUse1HooksRegistered();

}
