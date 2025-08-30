import logging
from typing import Callable, List, Union

from overrides import override

from ..machine import MachineConfig

from ..performance_model import OpInvokeInfo, PerformanceModel
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


def _get_op_estimator(
    op, machine_name
) -> Callable[[OpInvokeInfo, MachineConfig], PerformanceModel.Result]:
    if machine_name not in _op_estimator_table:
        machine_name = None
    if op not in _op_estimator_table[machine_name]:
        op = None
    return _op_estimator_table[machine_name][op]


def _estimate_default(
    op_invoke_info: OpInvokeInfo, machine_config: MachineConfig
) -> PerformanceModel.Result:
    perf_properties = op_invoke_info.get_perf_properties()
    # By default, we do not consider instruction-level parallelism when counting computation time
    compute_time_s = 0
    for dtype in MachineConfig.DTYPES:
        if dtype in perf_properties.compute_ops:
            if dtype not in machine_config.ops:
                logger.warning(
                    "Ignoring compute ops of %s for %s since it is not supported on %s",
                    dtype,
                    op_invoke_info,
                    machine_config,
                )
                continue
            compute_ops = perf_properties.compute_ops[dtype]
            machine_ops = machine_config.ops[dtype] * machine_config.compute_efficiency
            compute_time_s += compute_ops.fused_multiply_add_ops / machine_ops
            compute_time_s += compute_ops.arithmetic_ops / machine_ops
    memory_bandwidth = (
        machine_config.memory_bandwidth_bytes_ps * machine_config.memory_efficiency
    )
    memory_read_time_s = perf_properties.memory_read_bytes / memory_bandwidth
    memory_write_time_s = perf_properties.memory_write_bytes / memory_bandwidth
    memory_readwrite_time_s = perf_properties.memory_readwrite_bytes / memory_bandwidth
    memory_access_time_s = (
        memory_read_time_s + memory_write_time_s + memory_readwrite_time_s
    )
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


def _estimate_static_cost(
    op_invoke_info: OpInvokeInfo, machine_config: MachineConfig
) -> float:
    perf_properties = op_invoke_info.get_perf_properties()
    if (
        perf_properties.network_send_bytes > 0
        or perf_properties.network_receive_bytes > 0
    ):
        return 10 * 1e-6
    for dtype in MachineConfig.DTYPES:
        if dtype in perf_properties.compute_ops:
            if dtype not in machine_config.ops:
                continue
            compute_ops = perf_properties.compute_ops[dtype]
            if compute_ops.fused_multiply_add_ops > 0:
                return 5 * 1e-6
    return 2 * 1e-6


@register_op_estimator(None, "A2")
def _estimate_default_A2(
    op_invoke_info: OpInvokeInfo, machine_config: MachineConfig
) -> PerformanceModel.Result:
    if is_view_op(op_invoke_info.func):
        return PerformanceModel.Result(0.0)
    result = _estimate_default(op_invoke_info, machine_config)
    result.execution_time_s += _estimate_static_cost(op_invoke_info, machine_config)
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
