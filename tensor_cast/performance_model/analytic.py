import logging
from typing import Callable, Dict, List, Union
from overrides import override
import torch
import itertools

from ..performance_model import PerformanceModel, OpInvokeInfo
from ..machine import MachineConfig
from .utils import is_view_op

logger = logging.getLogger(__name__)

_op_estimator_table = {}

def register_op_estimator(op, machine_names: Union[str, List[str]]):
    if not isinstance(machine_names, (list, tuple)):
        machine_names = [machine_names]
    def decorator(estimator):
        for machine_name in machine_names:
            if machine_name not in _op_estimator_table:
                _op_estimator_table[machine_name] = {}
            assert op not in _op_estimator_table[machine_name]
            _op_estimator_table[machine_name][op] = estimator
        return estimator
    return decorator

def _get_op_estimator(op, machine_name) -> Callable[[OpInvokeInfo, MachineConfig], PerformanceModel.Result]:
    if machine_name not in _op_estimator_table:
        machine_name = None
    if op not in _op_estimator_table[machine_name]:
        op = None
    return _op_estimator_table[machine_name][op]

def _estimate_default(op_invoke_info: OpInvokeInfo, machine_config: MachineConfig) -> PerformanceModel.Result:
    perf_properties = op_invoke_info.get_perf_properties()
    # By default, we do not consider instruction-level parallelism when counting computation time
    compute_time_s = 0
    for dtype in MachineConfig.DTYPES:
        if dtype in perf_properties.compute_ops:
            if dtype not in machine_config.ops:
                logger.warning(f"Ignoring compute ops of {dtype} for {op_invoke_info} since it is not supported on {machine_config}")
                continue
            compute_ops = perf_properties.compute_ops[dtype]
            compute_time_s += compute_ops.fused_multiply_add_ops / machine_config.ops[dtype]
            compute_time_s += compute_ops.arithmetic_ops / machine_config.ops[dtype]
    memory_read_time_s = perf_properties.memory_read_bytes / machine_config.memory_bandwidth_bytes_ps
    memory_write_time_s = perf_properties.memory_write_bytes / machine_config.memory_bandwidth_bytes_ps
    memory_readwrite_time_s = perf_properties.memory_readwrite_bytes / machine_config.memory_bandwidth_bytes_ps
    memory_access_time_s = memory_read_time_s + memory_write_time_s + memory_readwrite_time_s
    time_s = max(compute_time_s, memory_access_time_s)
    result = PerformanceModel.Result(
        execution_time_s=time_s,
        statistics={
            "memory_read_time_s": memory_read_time_s,
            "memory_write_time_s": memory_write_time_s,
            "memory_readwrite_time_s": memory_readwrite_time_s,
            "memory_access_time_s": memory_access_time_s,
            "compute_time_s": compute_time_s,
        },
    )
    return result

@register_op_estimator(None, "910B")
def _estimate_default_A2(op_invoke_info: OpInvokeInfo, machine_config: MachineConfig) -> PerformanceModel.Result:
    if is_view_op(op_invoke_info.func):
        return PerformanceModel.Result(0.0)
    result = _estimate_default(op_invoke_info, machine_config)
    # XXX: what is the right static cost? 5us?
    static_cost = 5*1e-6
    result.execution_time_s += static_cost
    return result

register_op_estimator(None, None)(_estimate_default)

class AnalyticPerformanceModel(PerformanceModel):
    """
    Analytic performance model uses simple roofline model to estimate the
    op execution time.
    TODO: add cache model to more accurately estimate the execution time.
    """
    def __init__(self, machine_config: MachineConfig):
        super().__init__("analytic", machine_config)

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        op_estimator = _get_op_estimator(op_invoke_info.func, self.machine_config.name)
        result = op_estimator(op_invoke_info, self.machine_config)
        return result
