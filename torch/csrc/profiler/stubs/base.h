#pragma once

#include <functional>
#include <memory>

#include <c10/util/strong_type.h>
#include <torch/csrc/Export.h>

struct CUevent_st;

namespace torch {
namespace profiler {
namespace impl {

// ----------------------------------------------------------------------------
// -- Annotation --------------------------------------------------------------
// ----------------------------------------------------------------------------
using ProfilerEventStub = std::shared_ptr<CUevent_st>;
using ProfilerVoidEventStub = std::shared_ptr<void>;

struct ProfilerStubs {
  virtual void record(
      int* device,
      ProfilerVoidEventStub* event,
      int64_t* cpu_ns) const = 0;
  virtual float elapsed(
      const ProfilerVoidEventStub* event,
      const ProfilerVoidEventStub* event2) const = 0;
  virtual void mark(const char* name) const = 0;
  virtual void rangePush(const char* name) const = 0;
  virtual void rangePop() const = 0;
  virtual bool enabled() const {
    return false;
  }
  virtual void onEachDevice(std::function<void(int)> op) const = 0;
  virtual void synchronize() const = 0;
  virtual ~ProfilerStubs();
};

void registerCUDAMethods(ProfilerStubs* stubs);
const ProfilerStubs* cudaStubs();
void registerITTMethods(ProfilerStubs* stubs);
const ProfilerStubs* ittStubs();
void registerPrivateUse1Methods(ProfilerStubs* stubs);
const ProfilerStubs* privateuse1Stubs();

using vulkan_id_t = strong::type<
    int64_t,
    struct _VulkanID,
    strong::regular,
    strong::convertible_to<int64_t>,
    strong::hashable>;

} // namespace impl
} // namespace profiler
} // namespace torch
