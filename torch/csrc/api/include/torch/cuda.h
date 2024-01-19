#pragma once

#include <torch/csrc/Export.h>

#include <cstddef>
#include <cstdint>

namespace torch {
namespace cuda {

/// Returns the number of CUDA devices available.
size_t device_count();

/// Returns true if at least one CUDA device is available.
bool is_available();

/// Returns true if CUDA is available, and CuDNN is available.
bool cudnn_is_available();

/// Sets the seed for the current GPU.
void manual_seed(uint64_t seed);

/// Sets the seed for all available GPUs.
void manual_seed_all(uint64_t seed);

/// Waits for all kernels in all streams on a CUDA device to complete.
void synchronize(int64_t device_index = -1);

} // namespace cuda
} // namespace torch
