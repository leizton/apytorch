#pragma once

#include <c10/macros/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

/*
 * This file contains APIs for cuda fuser;
 *
 * We use an empty static struct to hold the function pointers, which are
 * registered separately. This is to support cpu-only compilation.
 * Registration is done in torch/csrc/jit/codegen/cuda/register_interface.cpp
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::atomic<bool>& getCudaFusionGuardMode();

bool getSingletonFusion();
bool setSingletonFusion(bool value);
bool getHorizontalFusion();
bool setHorizontalFusion(bool value);

// dummy struct to allow API registration
struct CudaFuserInterface {
  void (*fn_compile_n)(Node*) = nullptr;
  void (*fn_run_n_s)(const Node*, Stack&) = nullptr;
  void (*fn_fuse_graph)(std::shared_ptr<Graph>&) = nullptr;
  bool (*fn_can_fuse_n)(const Node*) = nullptr;
  void (*fn_insert_profile_inodes)(ProfilingRecord* pr) = nullptr;
  bool (*fn_profile_n)(const Node*) = nullptr;
  bool (*fn_skip_n)(const std::string&, bool flip) = nullptr;
};

// Get interface, this is used by registration and user facing API internally
CudaFuserInterface* getFuserInterface();

void compileFusionGroup(Node* fusion_node);
void runFusionGroup(const Node* fusion_node, Stack& stack);
void fuseGraph(std::shared_ptr<Graph>&);
bool canFuseNode(const Node* node);
void InsertProfileNodesForCUDAFuser(ProfilingRecord* pr);
bool profileNode(const Node* node);

bool skipNode(const std::string& symbol_str, bool flip = true);

bool isEnabled();
bool setEnabled(bool is_enabled);
bool canBeEnabled();

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
