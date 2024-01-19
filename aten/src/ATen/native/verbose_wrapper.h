#pragma once

#include <c10/macros/Export.h>

namespace torch::verbose {
int _mkl_set_verbose(int enable);
int _mkldnn_set_verbose(int level);
} // namespace torch::verbose
