#pragma once

#include <ATen/Utils.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

namespace c10 {

class DynamicLibraryError : public Error {
  using Error::Error;
};

} // namespace c10

namespace at {

struct DynamicLibrary {
  AT_DISALLOW_COPY_AND_ASSIGN(DynamicLibrary);

  DynamicLibrary(
      const char* name,
      const char* alt_name = nullptr,
      bool leak_handle = false);

  void* sym(const char* name);

  ~DynamicLibrary();

 private:
  bool leak_handle;
  void* handle = nullptr;
};

} // namespace at
