import dataclasses
import hashlib
import itertools
import logging

from typing import Dict, Optional

import torch

from .. import ops  # noqa: F401
from .utils import bytes_of_tensor, is_view_op, run_once

logger = logging.getLogger(__name__)


class OpInvokeInfo:
    _op_properties_functors = {}

    @dataclasses.dataclass
    class ComputeOps:
        mma_ops: int = 0
        """Number of Matrix-Multiply-Accumulate ops"""
        gp_ops: int = 0
        """Number of General-Purpose ops"""

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

        def combine(
            self, other: "OpInvokeInfo.PerformanceProperties", compute_only=False
        ):
            for dtype, compute_ops in other.compute_ops.items():
                if dtype not in self.compute_ops:
                    self.compute_ops[dtype] = OpInvokeInfo.ComputeOps()
                self.compute_ops[dtype].mma_ops += compute_ops.mma_ops
                self.compute_ops[dtype].gp_ops += compute_ops.gp_ops
            if not compute_only:
                self.memory_read_bytes += other.memory_read_bytes
                self.memory_write_bytes += other.memory_write_bytes
                self.memory_readwrite_bytes += other.memory_readwrite_bytes

    def __init__(self, func, args, kwargs, out, cache_key=None):
        self.func = func
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
        self.out = out
        self.cache_key = cache_key or self.compute_cache_key()

    @classmethod
    def get_op_properties_functor(cls, op):
        def default_functor(self: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
            """Default functor only counts in the memory accesses"""
            if is_view_op(self.func):
                return OpInvokeInfo.PerformanceProperties()
            run_once(
                self.func,
                logger.warning,
                f"No op properties function defined for {self.func}, "
                f"assuming it is memory-bandwidth bound.",
            )
            return self.get_memory_access_properties()

        if op not in OpInvokeInfo._op_properties_functors:
            return default_functor
        return OpInvokeInfo._op_properties_functors[op]

    @classmethod
    def register_op_properties(cls, op, override=False):
        def decorator(functor):
            if op in OpInvokeInfo._op_properties_functors:
                if override:
                    logger.warning(
                        "Overwriting existing properties functor for op: %s", op
                    )
                else:
                    raise ValueError(f"Op {op} already registered")
            OpInvokeInfo._op_properties_functors[op] = functor
            return functor

        return decorator

    def get_memory_access_properties(
        self,
        exclude_input_ids: Optional[set] = None,
        exclude_output_ids: Optional[set] = None,
    ) -> "OpInvokeInfo.PerformanceProperties":
        """Get memory read/write properties"""
        exclude_input_ids = set() if exclude_input_ids is None else exclude_input_ids
        exclude_output_ids = set() if exclude_output_ids is None else exclude_output_ids
        memory_read_bytes = 0
        memory_write_bytes = 0
        memory_readwrite_bytes = 0
        args_schema = self.func._schema.arguments
        for i, arg in enumerate(itertools.chain(self.args, self.kwargs.values())):
            if i not in exclude_input_ids:
                inputs = arg if isinstance(arg, (list, tuple)) else [arg]
                if inputs and isinstance(inputs[0], torch.Tensor):
                    for tensor in inputs:
                        access_bytes = bytes_of_tensor(tensor)
                        if args_schema[i].is_out:
                            memory_write_bytes += access_bytes
                        elif args_schema[i].is_write:
                            memory_readwrite_bytes += access_bytes
                        else:
                            memory_read_bytes += access_bytes
        out = self.out if isinstance(self.out, (list, tuple)) else [self.out]
        for i, arg in enumerate(out):
            if isinstance(arg, torch.Tensor) and i not in exclude_output_ids:
                access_bytes = bytes_of_tensor(arg)
                memory_write_bytes += access_bytes
        return OpInvokeInfo.PerformanceProperties(
            memory_read_bytes=memory_read_bytes,
            memory_write_bytes=memory_write_bytes,
            memory_readwrite_bytes=memory_readwrite_bytes,
        )

    def get_perf_properties(self) -> "OpInvokeInfo.PerformanceProperties":
        functor = self.get_op_properties_functor(self.func)
        return functor(self)

    def compute_cache_key(self) -> str:
        """
        Compute an efficient cache key based on operation signature and tensor properties.
        This key represents the computational characteristics of the operation.

        Returns:
            A string hash that can be used as a cache key
        """
        key_components = []

        key_components.append(str(self.func))

        def add_tensor_info(t, components):
            if isinstance(t, torch.Tensor):
                components.extend(
                    [
                        str(t.shape),
                        str(t.dtype),
                        str(t.device),
                        str(t.stride()) if not t.is_contiguous() else "contiguous",
                        str(t.requires_grad),
                    ]
                )
            else:
                components.append(str(t))

        for arg in self.args:
            add_tensor_info(arg, key_components)

        if self.kwargs:
            for k, v in sorted(self.kwargs.items()):
                key_components.append(k)
                add_tensor_info(v, key_components)

        # Create hash
        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def __repr__(self):
        return f"OpInvokeInfo({self.func}, {self.args}, {self.kwargs}, {self.out})"
