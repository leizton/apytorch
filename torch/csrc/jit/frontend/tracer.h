#pragma once

#include <ATen/core/Dimname.h>
#include <ATen/core/class_type.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <ATen/core/symbol.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>

#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/utils/variadic.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
struct Node;
struct Value;
struct Graph;
struct Module;

namespace tracer {

using ::c10::ivalue::Shared;

using ::c10::IValue;
using ::c10::ivalue::Future;

using ::c10::ArrayRef;
using ::c10::TupleType;
using ::c10::TupleTypePtr;
using ::c10::ivalue::ConstantString;

using torch::autograd::Variable;
using variable_list = std::vector<Variable>;

std::atomic<bool>& getTracerStateWarnMode();

struct TracingState
    : public std::enable_shared_from_this<TracingState> {
  TracingState();
  ~TracingState();

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<Graph> graph;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool warn = getTracerStateWarnMode();
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool strict = true;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool force_outplace = false;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::function<std::string(const Variable& var)> lookup_var_name_fn =
      [](const Variable& var) { return ""; };

  void enterFrame() {
    env_stack.emplace_back();
  }

  void leaveFrame() {
    env_stack.pop_back();
  }

  void setValue(const IValue& v, Value* value);
  void delValue(const IValue& var);
  Value* getValue(const IValue& var);
  Value* getOutput(const IValue& var, size_t i);
  bool hasValue(const IValue& var) const;

  Node* createNode(c10::Symbol op_name, size_t num_outputs);
  void insertNode(Node* node);

 private:
  using WeakIValue = at::WeakIValue;

  struct WeakIValueHasher {
    size_t operator()(const WeakIValue& t) const {
      return t.hash();
    }
  };

  struct WeakIValueEq {
    bool operator()(const WeakIValue& t1, const WeakIValue& t2) const {
      return t1.isSameIdentity(t2);
    }
  };

  using Frame =
      std::unordered_map<WeakIValue, Value*, WeakIValueHasher, WeakIValueEq>;
  std::vector<Frame> env_stack;
};

// This is meant to be used as a thread local place, where we can store extra
// info that gets lost when we call into ATen from Python bindings. One example
// for when this happens is when we get an IntArrayRef argument with e.g. sizes
// for view. When tracing, those might be tensors, which let us encode extra
// data dependencies, but once they get to the ATen call where we actually have
// the tracing logic, they get converted into a raw IntArrayRef, and we loose
// all information. To prevent this, we temporarily stash it in here.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct ArgumentStash {
  struct IntArrayRefTrace : std::vector<Value*> {
    IntArrayRefTrace(int size) : std::vector<Value*>(size, nullptr) {}
  };

  static bool empty() {
    return stash.intlists.empty();
  }

  static void stashIntArrayRefElem(
      const std::string& arg_name,
      size_t size,
      size_t idx,
      const Variable& var);

  static bool hasIntArrayRef(const std::string& arg_name) {
    return stash.intlists.count(arg_name) > 0;
  }

  static IntArrayRefTrace popIntArrayRef(const std::string& arg_name) {
    auto info = std::move(stash.intlists.at(arg_name));
    stash.intlists.erase(arg_name);
    return info;
  }

  // Value stashing: Use these methods to stash arguments which correspond
  // to regular Value*'s in the graph. i.e. they don't require special
  // handling like in the case of IntArrayRefs
  static void stashValue(
      const std::string& arg_name,
      size_t idx,
      const Variable& var,
      const c10::TypePtr& type = nullptr);

  static bool hasValue(const std::string& arg_name) {
    return stash.values.count(arg_name) > 0;
  }

  static Value* popValue(const std::string& arg_name) {
    auto info = stash.values.at(arg_name);
    stash.values.erase(arg_name);
    return info;
  }

 private:
  static thread_local ArgumentStash stash;
  std::unordered_map<std::string, IntArrayRefTrace> intlists;
  std::unordered_map<std::string, Value*> values;
};

// Retrieve or set the current tracing state. Returns a nullptr if tracing is
// disabled.
const std::shared_ptr<TracingState>& getTracingState();
void setTracingState(std::shared_ptr<TracingState> state);

inline bool isTracing() {
  return static_cast<bool>(getTracingState());
}

using warn_fn_type = void (*)(const std::string& msg);
extern const char* WARN_PYTHON_DATAFLOW;
extern const char* WARN_CONSTRUCTOR;
extern const char* WARN_RESIZE;
extern const char* STRICT_TRACER_MSG;
void _do_warn(const char* _reason, const char* _kind);
inline void warn(const char* _reason, const char* _kind = nullptr) {
  if (const auto& state = getTracingState()) {
    if (!state->warn)
      return;
    _do_warn(_reason, _kind);
  }
}
void setWarn(warn_fn_type fn);

struct NoWarn {
  NoWarn() : state(getTracingState()) {
    if (state) {
      prev = state->warn;
      state->warn = false;
    }
  }
  ~NoWarn() {
    if (state) {
      state->warn = prev;
    }
  }
  std::shared_ptr<TracingState> state;
  bool prev{false};
};

struct WithNestedTracingFrame {
  WithNestedTracingFrame() {
    getTracingState()->enterFrame();
  }

  ~WithNestedTracingFrame() {
    getTracingState()->leaveFrame();
  }
};
void recordSourceLocation(Node* n);
void setRecordSourceLocation(void (*v)(Node*));

std::vector<StackEntry> pythonCallstack();
void setPythonCallstack(std::vector<StackEntry> (*v)());

// Having finished adding a new 'node' to the graph IR 'setValueTrace'
// associates this node with an output variable, so that further operations
// involving this variable know which node in the IR to reference.
void setValueTrace(const IValue& v, Value* value);

void delValueTrace(const IValue& var);

std::function<void()> pauseTracing();

Value* getValueTrace(const IValue& var);

std::pair<std::shared_ptr<TracingState>, Stack> trace(
    Stack inputs,
    const std::function<Stack(Stack)>& traced_fn,
    std::function<std::string(const Variable&)> var_name_lookup_fn,
    bool strict = true,
    bool force_outplace = false,
    Module* self = nullptr,
    const std::vector<std::string>& argument_names = {});

void abandon();

// NB: those serve both as an intermediate steps in addInputs below,
// as well as the overloads that terminate template recursion
void addInputs(Node* n, const char* name, int64_t value);
void addInputs(Node* n, const char* name, c10::SymInt value);
void addInputs(
    Node* n,
    const char* name,
    c10::optional<int64_t> value);
void addInputs(Node* n, const char* name, bool value);
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<bool>& value);
void addInputs(Node* n, const char* name, double value);
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<double>& value);
void addInputs(Node* n, const char* name, const at::Scalar& value);
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Scalar>& value);
void addInputs(Node* n, const char* name, const at::Tensor& value);
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Tensor>& value);
void addInputs(Node* n, const char* name, ArrayRef<int64_t> value);
void addInputs(Node* n, const char* name, c10::SymIntArrayRef value);
void addInputs(
    Node* n,
    const char* name,
    c10::optional<c10::SymInt> value);
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<ArrayRef<int64_t>>& value);
void addInputs(
    Node* n,
    const char* name,
    const at::OptionalIntArrayRef& opt_value);
void addInputs(
    Node* n,
    const char* name,
    const at::OptionalSymIntArrayRef& opt_value);
void addInputs(
    Node* n,
    const char* name,
    ArrayRef<at::Tensor> value,
    bool allow_undefined = false);
void addInputs(
    Node* n,
    const char* name,
    std::vector<at::Tensor> value,
    bool allow_undefined = false);
void addInputs(
    Node* n,
    const char* name,
    at::ITensorListRef value,
    bool allow_undefined = false);
void addInputs(
    Node* n,
    const char* name,
    const List<c10::optional<at::Tensor>>& value);
void addInputs(
    Node* n,
    const char* name,
    ArrayRef<c10::intrusive_ptr<c10::ivalue::Object>> value,
    const c10::ClassTypePtr& class_type);
void addInputs(Node* n, const char* name, ArrayRef<double> value);
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<ArrayRef<double>>& value);
void addInputs(
    Node* n,
    const char* name,
    const c10::string_view value);
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<c10::string_view>& value);
void addInputs(Node* n, const char* name, at::Device value);
void addInputs(Node* n, const char* name, c10::Stream stream);
void addInputs(Node* n, const char* name, at::Layout value);
void addInputs(Node* n, const char* name, at::ScalarType value);
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::ScalarType>& value);
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Device>& value);
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Layout>& value);
void addInputs(Node* n, const char* name, at::MemoryFormat value);
void addInputs(
    Node* n,
    const char* name,
    c10::optional<at::DimnameList> value);
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::MemoryFormat>& value);
void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Generator>& value);

inline void addInputs(
    Node* n,
    const char* name,
    const std::vector<bool>& value) {
  AT_ERROR("Tracing a list of bool type is currently not supported!");
}

template <typename T>
void addInputs(Node* n, const char* name, ArrayRef<T> value) {
  AT_ERROR("Tracing a list of arbitrary type is currently not supported!");
}
template <typename K, typename V>
void addInputs(
    Node* n,
    const char* name,
    const std::unordered_map<K, V>& value) {
  AT_ERROR("Tracing a dict of arbitrary types is currently not supported!");
}

template <size_t N>
void addInputs(Node* n, const char* name, std::array<bool, N> value) {
  throw std::runtime_error(
      "Found an unsupported argument type in the JIT tracer. File a bug report.");
}

void addInputs(
    Node* n,
    const char* name,
    const c10::intrusive_ptr<c10::ivalue::Object>& obj);

void ensureUniqueIfOutOfPlaced(
    const char* name,
    const at::Tensor& tensor);
void ensureUniqueIfOutOfPlaced(
    const char* name,
    const c10::optional<at::Tensor>& tensor);

template <
    typename T,
    typename = torch::enable_if_t<(
        !std::is_convertible<torch::decay_t<T>, at::TensorList>::value &&
        !std::is_convertible<torch::decay_t<T>, c10::List<at::Tensor>>::value &&
        !std::is_convertible<torch::decay_t<T>, at::Tensor>::value &&
        !std::is_convertible<
            torch::decay_t<T>,
            c10::intrusive_ptr<c10::ivalue::Object>>::value)>>
void addOutput(Node* node, T&&) {
  AT_ERROR(
      "Found an unsupported argument type ",
      c10::demangle_type<T>(),
      " in the JIT tracer. File a bug report.");
}
void addOutput(Node* node, const at::Tensor& tensor);
void setOutput(Value* value, const at::Tensor& output);
void addOutput(Node* node, const std::vector<at::Tensor>& list);
void addOutput(Node* node, const c10::List<at::Tensor>& list);
void addOutput(
    Node* node,
    const c10::intrusive_ptr<c10::ivalue::Object>& output);

autograd::Variable getSizeOf(
    const autograd::Variable& var,
    int64_t dim);

autograd::Variable getNumelOf(const autograd::Variable& var);

} // namespace tracer
} // namespace jit
} // namespace torch
