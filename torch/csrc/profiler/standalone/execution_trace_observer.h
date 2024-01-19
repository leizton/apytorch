#pragma once

#include <c10/macros/Export.h>
#include <string>

namespace torch {
namespace profiler {
namespace impl {

// Adds the execution trace observer as a global callback function, the data
// will be written to output file path.
bool addExecutionTraceObserver(const std::string& output_file_path);

// Remove the execution trace observer from the global callback functions.
void removeExecutionTraceObserver();

// Enables execution trace observer.
void enableExecutionTraceObserver();

// Disables execution trace observer.
void disableExecutionTraceObserver();

} // namespace impl
} // namespace profiler
} // namespace torch
