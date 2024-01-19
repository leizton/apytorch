#pragma once

#include <torch/csrc/lazy/core/ir.h>

namespace torch {
namespace lazy {

NodePtr operator+(const Value& node1, const Value& node2);
NodePtr operator-(const Value& node1, const Value& node2);
NodePtr operator*(const Value& node1, const Value& node2);
NodePtr operator/(const Value& node1, const Value& node2);

} // namespace lazy
} // namespace torch
