#pragma once
#include <ATen/core/TorchDispatchUtils.h>

namespace at {
namespace impl {

struct RestorePythonTLSSnapshot {
  RestorePythonTLSSnapshot();
  ~RestorePythonTLSSnapshot();

private:
  c10::impl::LocalDispatchKeySet saved_;
  c10::impl::ForceDispatchKeyGuard guard_;
};


// RAII guard to make working with the above TLS safer.
struct MaybeSetTLSOnEntryGuard {
public:
  MaybeSetTLSOnEntryGuard();
  ~MaybeSetTLSOnEntryGuard();

private:
  bool value_set_;
};

} // namespace impl
} // namespace at
