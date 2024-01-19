import math
import operator
import traceback
from functools import partial
from typing import Callable, Dict, List, NamedTuple, Set, Tuple

import sympy

import torch
import torch.fx
from torch._export.pass_base import _ExportPassBase, ProxyValue, PassResult
from torch.utils._sympy.value_ranges import ValueRanges


__all__ = ["InputDim"]


class InputDim(NamedTuple):
    input_name: str
    dim: int


def _convert_to_int(val):
    # Convert simple sympy Integers into concrete int
    if val == sympy.oo:
        return math.inf
    if val == -sympy.oo:
        return -math.inf
    if isinstance(val, sympy.Integer):
        return int(val)
    raise RuntimeError(
        "Export constraints cannot be non-integer expressions"
    )


def _convert_range_to_int(range: ValueRanges):
    assert isinstance(range, ValueRanges)
    min_val = _convert_to_int(range.lower)
    max_val = _convert_to_int(range.upper)
    return min_val, max_val


class _AddRuntimeAssertionsForInlineConstraintsPass(_ExportPassBase):
    def __init__(
        self,
        range_constraints: Dict[sympy.Symbol, ValueRanges],
        equality_constraints: List[Tuple[InputDim, InputDim]],
    ):
        super().__init__()
        self.range_constraints: Dict[sympy.Symbol, ValueRanges] = range_constraints
        self.equality_constraints: List[Tuple[InputDim, InputDim]] = equality_constraints
        self._asserts_generated_unbacked_symbols: Set[sympy.Symbol] = set()
        self.counter = 0

    def _assert_range_constraint(self, proxy, lower, upper, assert_msg):
        if lower > -math.inf:
            self._insert_assert_async(operator.ge, proxy, lower, assert_msg)

        if upper < math.inf:
            self._insert_assert_async(operator.le, proxy, upper, assert_msg)

    def _insert_assert_async(self, operator, lower, upper, assert_msg):
        """
        Inserts assert_async call_function nodes in the graph. This function is
        called **during** the interpreter-based pass.
        """
        self.counter += 1
        cmp = super().call_operator(operator, (lower, upper), {}, self._create_dummy_node_metadata())
        cmp_tensor = super().call_operator(torch.ops.aten.scalar_tensor.default, (cmp,), {}, self._create_dummy_node_metadata())
        super().call_operator(
            torch.ops.aten._assert_async.msg,
            (cmp_tensor, assert_msg),
            {},
            self._create_dummy_node_metadata(),
        )

    def call_operator(self, op, args, kwargs, meta) -> ProxyValue:
        ret = super().call_operator(op, args, kwargs, meta)
        if "val" not in meta:
            return ret

        val = meta["val"]

        # In general, we may have to deal the case such as: ret[1].shape[0].
        # We need first find out what symbols require assertion, then we need to follow the path
        # from ret to the symbol, construct the proxies along the way and construct the messages
        # piece-wise at the same time.
        #
        # We use post-order traversal to collect all the proxies callbacks needed, construct
        # the error message callbacks, and at the top-level traversal tree we execute all the callbacks.
        # We need the callbacks because, in order to call the function to create a proxy for shape[0], we
        # need the proxy for shape, which further requires the proxy for ret[1], etc.
        def add_assertions(val):
            call_backs: List[Callable] = []
            messages: List[str] = []
            if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                symbol = val.node._expr
                if isinstance(symbol, sympy.Symbol) and symbol.name.startswith("i"):
                    if symbol in self._asserts_generated_unbacked_symbols:
                        return call_backs, messages
                    # We only care about unbacked symints for these inline
                    # constraints, which are prefixed with 'i'
                    constraint = self.range_constraints[symbol]
                    min_val, max_val = _convert_range_to_int(constraint)
                    assert_msg = f" is outside of inline constraint [{min_val}, {max_val}]."
                    call_backs.append(
                        partial(self._assert_range_constraint, lower=min_val, upper=max_val)
                    )
                    messages.append(assert_msg)
                    self._asserts_generated_unbacked_symbols.add(symbol)
            elif isinstance(val, torch.Tensor):
                for i, sym in enumerate(val.shape):
                    cbs, msgs = add_assertions(sym)
                    for cb, msg in zip(cbs, msgs):
                        def sym_size_cb(proxy, assert_msg, dim):
                            dim_proxy = super(
                                _AddRuntimeAssertionsForInlineConstraintsPass,
                                self
                            ).call_operator(
                                torch.ops.aten.sym_size.int,
                                (proxy, dim),
                                {},
                                self._create_dummy_node_metadata(),
                            )
                            cb(proxy=dim_proxy, assert_msg=assert_msg)
                        call_backs.append(partial(sym_size_cb, dim=i))
                        messages.append(f".shape[{i}]" + msg)
            return call_backs, messages
        callbacks, messages = add_assertions(val)
        for cb, msg in zip(callbacks, messages):
            cb(proxy=ret, assert_msg=f"{ret.node}" + msg)
        return ret

    def call(self, graph_module):
        # Add runtime asserts for inline constraints
        val = super().call(graph_module)

        # Sometimes this pass would return a wrong graph where we have mismatched
        # node names in signature. Before we fix it, let's just skip it.
        if self.counter == 0 and type(self) is _AddRuntimeAssertionsForInlineConstraintsPass:
            return PassResult(graph_module, False)

        # Populate the stack trace with dummy vals to respect IR
        for node in val.graph_module.graph.nodes:
            if not node.meta.get("stack_trace", None):
                node.meta["stack_trace"] = "".join(traceback.format_stack(limit=1))

        return PassResult(val.graph_module, val.modified)