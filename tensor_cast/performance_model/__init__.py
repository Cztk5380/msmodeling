import dataclasses
import itertools
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

from ..machine import MachineConfig
from .utils import is_view_op, run_once

logger = logging.getLogger(__name__)


class OpInvokeInfo:
    _op_properties_functors = {}

    @dataclasses.dataclass
    class ComputeOps:
        fused_multiply_add_ops: int = 0
        arithmetic_ops: int = 0

    @dataclasses.dataclass
    class PerformanceProperties:
        compute_ops: Dict[torch.dtype, "OpInvokeInfo.ComputeOps"] = dataclasses.field(
            default_factory=dict
        )
        memory_read_bytes: int = 0
        """Read-only bytes"""
        memory_write_bytes: int = 0
        """Write-only bytes"""
        memory_readwrite_bytes: int = 0
        """Read-write bytes"""
        network_send_bytes: int = 0
        network_receive_bytes: int = 0

    def __init__(self, func, args, kwargs, out):
        self.func = func
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
        self.out = out

    @classmethod
    def get_op_properties_functor(cls, op):
        def default_functor(self: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
            """Default functor only counts in the memory accesses"""
            if is_view_op(self.func):
                return OpInvokeInfo.PerformanceProperties()
            run_once(
                self.func,
                logger.warning,
                f"No performance estimator defined for {self.func}, "
                f"assuming it is memory-bandwidth bound.",
            )
            return self.get_memory_access_properties()

        if op not in OpInvokeInfo._op_properties_functors:
            return default_functor
        return OpInvokeInfo._op_properties_functors[op]

    @classmethod
    def register_op_properties(cls, op):
        def decorator(functor):
            assert op not in OpInvokeInfo._op_properties_functors, (
                f"Op properties functor for {op} already registered"
            )
            OpInvokeInfo._op_properties_functors[op] = functor
            return functor

        return decorator

    def get_memory_access_properties(self) -> "OpInvokeInfo.PerformanceProperties":
        """Get memory read/write properties"""
        memory_read_bytes = 0
        memory_write_bytes = 0
        memory_readwrite_bytes = 0
        args_schema = self.func._schema.arguments
        for i, arg in enumerate(itertools.chain(self.args, self.kwargs.values())):
            if isinstance(arg, torch.Tensor):
                access_bytes = arg.element_size() * arg.nelement()
                if args_schema[i].is_out:
                    memory_write_bytes += access_bytes
                elif args_schema[i].is_write:
                    memory_readwrite_bytes += access_bytes
                else:
                    memory_read_bytes += access_bytes
        out = self.out if isinstance(self.out, (list, tuple)) else [self.out]
        for arg in out:
            if isinstance(arg, torch.Tensor):
                access_bytes = arg.element_size() * arg.nelement()
                memory_write_bytes += access_bytes
        return OpInvokeInfo.PerformanceProperties(
            memory_read_bytes=memory_read_bytes,
            memory_write_bytes=memory_write_bytes,
            memory_readwrite_bytes=memory_readwrite_bytes,
        )

    def get_perf_properties(self) -> "OpInvokeInfo.PerformanceProperties":
        functor = self.get_op_properties_functor(self.func)
        return functor(self)

    def __repr__(self):
        return f"OpInvokeInfo({self.func}, {self.args}, {self.kwargs}, {self.out})"


class PerformanceModel(ABC):
    """
    Performance model used to estimate the execution time of op invocations
    on a given machine.
    """

    @dataclasses.dataclass
    class Result:
        execution_time_s: float
        statistics: Dict[str, Any] = dataclasses.field(default_factory=dict)
        """Misc runtime statistics produced by implementation"""

    def __init__(self, name, machine_config: MachineConfig):
        self.name = name
        self.machine_config = machine_config

    @abstractmethod
    def process_op(self, op_invoke_info: OpInvokeInfo) -> "PerformanceModel.Result":
        """
        Estimate the execution time of an op invocation on the given machine.
        Returns:
            op execution time in seconds and misc runtime statistics
        """


@OpInvokeInfo.register_op_properties(torch.ops.aten.bmm.default)
def _bmm_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 2
    mat1 = op_invoke_info.args[0]
    mat2 = op_invoke_info.args[1]
    assert isinstance(mat1, torch.Tensor)
    assert isinstance(mat2, torch.Tensor)
    assert mat1.ndim == 3
    assert mat2.ndim == 3
    b = mat1.size(0)
    m = mat1.size(1)
    k = mat1.size(2)
    n = mat2.size(2)
    assert mat2.size(0) == b
    assert mat2.size(1) == k
    properties = op_invoke_info.get_memory_access_properties()
    properties.compute_ops[mat1.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[mat1.dtype].fused_multiply_add_ops = b * m * n * k * 2
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.aten.mm.default)
def _mm_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 2
    mat1 = op_invoke_info.args[0]
    mat2 = op_invoke_info.args[1]
    assert isinstance(mat1, torch.Tensor)
    assert isinstance(mat2, torch.Tensor)
    assert mat1.ndim == 2
    assert mat2.ndim == 2
    m = mat1.size(0)
    k = mat1.size(1)
    n = mat2.size(1)
    assert mat2.size(0) == k
    properties = op_invoke_info.get_memory_access_properties()
    properties.compute_ops[mat1.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[mat1.dtype].fused_multiply_add_ops = m * n * k * 2
    return properties
