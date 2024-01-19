#include <torch/csrc/autograd/python_engine.h>

#include <ATen/LegacyBatchedTensorImpl.h>
#include <ATen/LegacyVmapMode.h>
#include <c10/util/irange.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_saved_variable_hooks.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>

#ifndef _WIN32
#include <pthread.h>
#endif

#include <memory> // for unique_ptr
#include <unordered_set>
#include <utility>

using namespace torch::autograd;

struct THPEngine {
  PyObject_HEAD
};

static bool _reinitialize_engine = false;

namespace torch {
namespace autograd {
namespace python {

PythonEngine::PythonEngine() = default;

Engine& PythonEngine::get_python_engine() {
  static PythonEngine engine;
  // This is "probably" thread-safe because the flag is set in a fork handler
  // before any threads are created, and this function is only called with the
  // GIL held. However, using fork + threads is playing with fire so this is
  // more of a "best effort" thing. For example, if the fork occurs while the
  // backwards threads hold a lock, we'll probably deadlock in the engine
  // destructor.
  if (_reinitialize_engine) {
    engine.release_workers();
    engine.~PythonEngine();
    new (&engine) torch::autograd::python::PythonEngine();
    _reinitialize_engine = false;
  }
  return engine;
}

PythonEngine::~PythonEngine() {
  Engine::stop();
}

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 9
#define IS_PYTHON_3_9_PLUS
#endif

void PythonEngine::thread_init(
    int device,
    const std::shared_ptr<ReadyQueue>& ready_queue,
    bool should_increment) {
  // Increment thread usage count before acquiring the GIL
  if (should_increment) {
    increment_non_reentrant_thread_count();
  }
  // Create a PyThreadState, but release the GIL. This lets
  // pybind11::gil_scoped_acquire calls inside thread_main acquire the GIL
  // without having to create a new PyThreadState each time.
#if defined(IS_PYTHON_3_9_PLUS)
  auto gil = std::make_unique<pybind11::gil_scoped_acquire>();
#else
  pybind11::gil_scoped_acquire gil;
#endif
  pybind11::gil_scoped_release no_gil;
  Engine::thread_init(device, ready_queue, false);

  if (should_increment) {
    // Decrement the count during shutdown if we incremented earlier.
    decrement_non_reentrant_thread_count();
  }

#if defined(IS_PYTHON_3_9_PLUS)
  // Do not call PyEval_RestoreThread, PyThreadState_[Clear|DeleteCurrent] if
  // runtime is finalizing
  if (!Py_IsInitialized()) {
    no_gil.disarm();
    // TODO: call disarm once PyThreadState_Clear can safely be called from
    // finalize NOTE: deploy.cpp calls `PyInterpreterState_Delete` to destruct
    // PyThreadState, so avoid use-after-free here.
    auto ptr = gil.release();
    operator delete(ptr);
  }
#endif
}

void PythonEngine::thread_on_exception(
    std::shared_ptr<GraphTask> graph_task,
    const std::shared_ptr<Node>& fn,
    std::exception& e) {
  // See Note [ Persisting PyErr state across autograd engine threads ]
  auto python_err = dynamic_cast<python_error*>(&e);
  if (python_err) {
    python_err->persist();
  }
  Engine::thread_on_exception(std::move(graph_task), fn, e);
}

std::unique_ptr<AnomalyMetadata> PythonEngine::make_anomaly_metadata() {
  return std::unique_ptr<AnomalyMetadata>(new PyAnomalyMetadata());
}

std::unique_ptr<SavedVariableHooks> PythonEngine::
    get_default_saved_variable_hooks() {
  return PyDefaultSavedVariableHooks::get_hooks();
}

variable_list PythonEngine::execute(
    const edge_list& roots,
    const variable_list& inputs,
    bool keep_graph,
    bool create_graph,
    bool accumulate_grad,
    const edge_list& outputs) {
  return Engine::execute(roots, inputs, keep_graph, create_graph, accumulate_grad, outputs);
}

c10::intrusive_ptr<at::ivalue::Future> PythonEngine::execute_with_graph_task(
    const std::shared_ptr<GraphTask>& graph_task,
    std::shared_ptr<Node> graph_root,
    InputBuffer&& input_buffer) {
  try {
    return Engine::execute_with_graph_task(
        graph_task, std::move(graph_root), std::move(input_buffer));
  } catch (python_error& e) {
    pybind11::gil_scoped_acquire gil;
    if (!PyErr_Occurred()) {
      // Set the error indicator only if it is not set already.
      e.restore();
    }
    throw;
  }
}
} // namespace python
} // namespace autograd
} // namespace torch

PyObject* THPEngineClass = nullptr;

// Implementation of torch._C._EngineBase.run_backward
PyObject* THPEngine_run_backward(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  PyObject* tensors = nullptr;
  PyObject* grad_tensors = nullptr;
  unsigned char keep_graph = 0;
  unsigned char create_graph = 0;
  PyObject* inputs = nullptr;
  unsigned char allow_unreachable = 0;
  unsigned char accumulate_grad = 0; // Indicate whether to accumulate grad into leaf Tensors or capture
  constexpr const char* accepted_kwargs[] = {// NOLINT
                                             "tensors",
                                             "grad_tensors",
                                             "keep_graph",
                                             "create_graph",
                                             "inputs",
                                             "allow_unreachable",
                                             "accumulate_grad",
                                             nullptr};

  Py_ssize_t num_tensors = PyTuple_GET_SIZE(tensors);
  Py_ssize_t num_gradients = PyTuple_GET_SIZE(grad_tensors);

  // The user either called autograd.backward(...) or autograd.grad(...) to get
  // here
  bool backward_api_called = accumulate_grad;

  edge_list roots;
  roots.reserve(num_tensors);
  variable_list grads;
  grads.reserve(num_tensors);
  for (const auto i : c10::irange(num_tensors)) {
    PyObject* _tensor = PyTuple_GET_ITEM(tensors, i);
    const auto& variable = THPVariable_Unpack(_tensor);
    auto gradient_edge = torch::autograd::impl::gradient_edge(variable);
    roots.push_back(std::move(gradient_edge));

    PyObject* grad = PyTuple_GET_ITEM(grad_tensors, i);
    const Variable& grad_var = THPVariable_Unpack(grad);
    grads.push_back(grad_var);
  }

  std::vector<Edge> output_edges;
  variable_list outputs;
  auto& engine = python::PythonEngine::get_python_engine();
  outputs = engine.execute(roots, grads, keep_graph, create_graph, accumulate_grad, output_edges);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPEngine_queue_callback(PyObject* self, PyObject* _callback) {
  HANDLE_TH_ERRORS
  auto& engine = python::PythonEngine::get_python_engine();
  std::shared_ptr<PyObject> callback(_callback, [](PyObject* obj) {
    pybind11::gil_scoped_acquire gil;
    Py_DECREF(obj);
  });
  Py_INCREF(_callback);
  engine.queue_callback([callback]() {
    pybind11::gil_scoped_acquire gil;
    THPObjectPtr result{PyObject_CallFunctionObjArgs(callback.get(), nullptr)};
    if (!result) {
      // Note [ Persisting PyErr state across autograd engine threads ]
      //
      // Since the autograd engine is multi-threaded, and Python error state is
      // local to each thread, it must preserve the python error from the worker
      // thread and rethrow it as-is in the calling thread. This is done via
      // persisting the error in the two places that can encounter Python
      // errors: (1) evaluate function and (2) queued callbacks.
      //
      // TODO: the engine is not actually responsible for persisting the error
      // in the custom autograd Function case today! See the note above
      // `raise_python_error()` function in python_function.cpp and
      // python_hooks.cpp for more details. Persisting an extra time in the
      // engine is fine because doing so is a no-op when the python_error has
      // already been persisted.
      python_error err;
      err.persist();
      throw std::move(err);
    }
  });
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPEngine_is_checkpoint_valid(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto& engine = python::PythonEngine::get_python_engine();
  if (engine.is_checkpoint_valid()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEngine_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  return type->tp_alloc(type, 0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyMethodDef THPEngine_methods[] = {
    {(char*)"run_backward",
     castPyCFunctionWithKeywords(THPEngine_run_backward),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {(char*)"queue_callback", THPEngine_queue_callback, METH_O, nullptr},
    {(char*)"is_checkpoint_valid",
     THPEngine_is_checkpoint_valid,
     METH_NOARGS,
     nullptr},
    {nullptr}};

PyTypeObject THPEngineType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._EngineBase", /* tp_name */
    sizeof(THPEngine), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THPEngine_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPEngine_new /* tp_new */
};

static void child_atfork() {
  _reinitialize_engine = true;
}

bool THPEngine_initModule(PyObject* module) {
#ifndef _WIN32
  if (pthread_atfork(nullptr, nullptr, child_atfork) != 0) {
    throw std::runtime_error("unable to set pthread_atfork handler");
  }
#endif
  if (PyType_Ready(&THPEngineType) < 0)
    return false;
  Py_INCREF(&THPEngineType);
  PyModule_AddObject(module, "_ImperativeEngine", (PyObject*)&THPEngineType);
  set_default_engine_stub(python::PythonEngine::get_python_engine);
  return true;
}
