#include <ATen/Context.h>

namespace at {

/// Returns a detailed string describing the configuration PyTorch.
std::string show_config();

std::string get_mkl_version();

std::string get_mkldnn_version();

std::string get_openmp_version();

std::string get_cxx_flags();

std::string get_cpu_capability();

} // namespace at
