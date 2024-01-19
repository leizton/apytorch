/**
 * This file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/master/third_party/xla_client/metrics.h
 */

#pragma once

#include <functional>
#include <memory>
#include <thread>

#include <c10/macros/Export.h>

namespace torch {
namespace lazy {

class Completion {
 public:
  class Data;

  explicit Completion(std::shared_ptr<Data> data);

  ~Completion();

  void Wait();

 private:
  std::shared_ptr<Data> data_;
};

// Schedules a closure which might wait for IO or other events/conditions.
void ScheduleIoClosure(std::function<void()> closure);
Completion
ScheduleIoClosureWithCompletion(std::function<void()> closure);

} // namespace lazy
} // namespace torch
