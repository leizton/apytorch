#pragma once

#include <c10/util/Optional.h>

#include <string>
#include <vector>

namespace torch {
namespace lazy {
struct SourceLocation {
  std::string file;
  std::string function;
  int line = -1;
};

void EmitShortFrameInfo(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames);

std::ostream& operator<<(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames);

// The base class for user defined metadata which is possible to attach to IR
// nodes.
struct UserMetaData {
  virtual ~UserMetaData() = default;
};

struct MetaData {
  std::string scope;
  std::vector<SourceLocation> frame_info;
};

// TODO(whc) is this going to be used outside of in IR decompositions?
// RAII data structure to be used a stack variable to enter a new IR scope. IR
// scope names will appear in the IR and will help identifying the source of the
// single IR nodes.
struct ScopePusher {
  explicit ScopePusher(const std::string& name);
  ~ScopePusher();

  static void ResetScopes();
};

MetaData GetMetaDataIfDebugging();

} // namespace lazy
} // namespace torch
