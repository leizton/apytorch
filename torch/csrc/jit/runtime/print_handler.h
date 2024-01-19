#pragma once

#include <torch/csrc/Export.h>

#include <string>

namespace torch::jit {

using PrintHandler = void (*)(const std::string&);

PrintHandler getDefaultPrintHandler();
PrintHandler getPrintHandler();
void setPrintHandler(PrintHandler ph);

} // namespace torch::jit
