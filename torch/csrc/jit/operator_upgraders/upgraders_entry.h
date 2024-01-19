#pragma once
#include <c10/macros/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <string>
#include <unordered_map>

namespace torch::jit {

void populate_upgraders_graph_map();

std::unordered_map<std::string, std::shared_ptr<Graph>>
generate_upgraders_graph();

std::unordered_map<std::string, std::string> get_upgraders_entry_map();

std::shared_ptr<Graph> create_upgrader_graph(
    const std::string& upgrader_name,
    const std::string& upgrader_body);

} // namespace torch::jit
