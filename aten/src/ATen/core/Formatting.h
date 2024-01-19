#pragma once

#include <ostream>
#include <string>

#include <c10/core/Scalar.h>
#include <ATen/core/Tensor.h>

namespace c10 {
std::ostream& operator<<(std::ostream& out, Backend b);
std::ostream& operator<<(std::ostream & out, const Scalar& s);
std::string toString(const Scalar& s);
}
namespace at {

std::ostream& operator<<(std::ostream& out, const DeprecatedTypeProperties& t);
std::ostream& print(
    std::ostream& stream,
    const Tensor& tensor,
    int64_t linesize);
static inline std::ostream& operator<<(std::ostream & out, const Tensor & t) {
  return print(out,t,80);
}
void print(const Tensor & t, int64_t linesize=80);
}
