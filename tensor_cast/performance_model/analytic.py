import logging
from enum import StrEnum
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from overrides import override

from ..device import DeviceProfile

from ..performance_model import OpInvokeInfo, PerformanceModel

from .utils import is_view_op

logger = logging.getLogger(__name__)


class StatsKey(StrEnum):
    COMPUTE = "compute_time_s"
    MMA_OPS = "mma_ops_time_s"
    GP_OPS = "gp_ops_time_s"
    MEMORY_ACCESS = "memory_access_time_s"
    COMMUNICATION = "comm_time_s"


_op_estimator_table = {}


def register_op_estimator(op, device_names: Optional[Union[str, List[str]]]):
    if not isinstance(device_names, (list, tuple)):
        device_names = [device_names]

    def decorator(estimator):
        for device_name in device_names:
            if device_name not in _op_estimator_table:
                _op_estimator_table[device_name] = {}
            assert op not in _op_estimator_table[device_name]
            _op_estimator_table[device_name][op] = estimator
        return estimator

    return decorator


def _get_op_estimator(
    op, device_name
) -> Callable[[OpInvokeInfo, DeviceProfile], PerformanceModel.Result]:
    if device_name not in _op_estimator_table:
        device_name = None
    if op not in _op_estimator_table[device_name]:
        op = None
    return _op_estimator_table[device_name][op]


class OpBoundClassifier(PerformanceModel.OpClassifier):
    @property
    def name(self):
        return "OpBound"

    def classify(
        self, event_list: List[Tuple[OpInvokeInfo, "PerformanceModel.Result"]]
    ) -> Dict[str, float]:
        COMPUTE_BOUND_MMA = "compute_bound_mma"
        COMPUTE_BOUND_GP = "compute_bound_gp"
        MEMORY_BOUND = "memory_bound"
        COMM_BOUND = "communication_bound"
        breakdown: Dict[str, float] = {
            MEMORY_BOUND: 0,
            COMM_BOUND: 0,
            COMPUTE_BOUND_MMA: 0,
            COMPUTE_BOUND_GP: 0,
        }
        breakdown_keys = list(breakdown.keys())
        for _, result in event_list:
            time_list = [
                result.statistics.get(StatsKey.MEMORY_ACCESS, 0),
                result.statistics.get(StatsKey.COMMUNICATION, 0),
                result.statistics.get(StatsKey.COMPUTE, 0),
            ]
            max_value = max(time_list)
            max_index = time_list.index(max_value)
            if max_index < 2:
                breakdown[breakdown_keys[max_index]] += max_value
            else:
                breakdown[COMPUTE_BOUND_MMA] += result.statistics.get(
                    StatsKey.MMA_OPS, 0
                )
                breakdown[COMPUTE_BOUND_GP] += result.statistics.get(StatsKey.GP_OPS, 0)
        return breakdown


class AnalyticPerformanceModel(PerformanceModel):
    """
    Analytic performance model uses simple roofline model to estimate the
    op execution time.
    TODO: add cache model to more accurately estimate the execution time.
    """

    def __init__(self, device_profile: DeviceProfile):
        super().__init__("analytic", device_profile)
        self.classifiers = [OpBoundClassifier()]

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        op_estimator = _get_op_estimator(op_invoke_info.func, self.device_profile.name)
        result = op_estimator(op_invoke_info, self.device_profile)
        return result

    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        return self.classifiers


def _estimate_static_cost(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> float:
    perf_properties = op_invoke_info.get_perf_properties()
    for dtype in DeviceProfile.DTYPES:
        if dtype in perf_properties.compute_ops:
            if dtype not in device_profile.mma_ops:
                continue
            compute_ops = perf_properties.compute_ops[dtype]
            if compute_ops.mma_ops > 0:
                return device_profile.static_cost.mma_op_cost_s
    return device_profile.static_cost.gp_op_cost_s


def _estimate_default_without_static_cost(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
    if is_view_op(op_invoke_info.func):
        return PerformanceModel.Result(0.0)
    perf_properties = op_invoke_info.get_perf_properties()
    # By default, we do not consider instruction-level parallelism when counting computation time
    mma_ops_time_s = 0
    gp_ops_time_s = 0
    for dtype in DeviceProfile.DTYPES:
        if dtype in perf_properties.compute_ops:
            compute_ops = perf_properties.compute_ops[dtype]
            if compute_ops.mma_ops > 0:
                if dtype in device_profile.mma_ops:
                    device_mma_ops = (
                        device_profile.mma_ops[dtype]
                        * device_profile.compute_efficiency
                    )
                    mma_ops_time_s += compute_ops.mma_ops / device_mma_ops
                else:
                    logger.warning(
                        "Ignoring mma compute ops of %s for %s since it is not supported on %s",
                        dtype,
                        op_invoke_info,
                        device_profile,
                    )
            if compute_ops.gp_ops > 0:
                if dtype in device_profile.gp_ops:
                    compute_ops = perf_properties.compute_ops[dtype]
                    device_gp_ops = (
                        device_profile.gp_ops[dtype] * device_profile.compute_efficiency
                    )
                    gp_ops_time_s += compute_ops.gp_ops / device_gp_ops
                else:
                    logger.warning(
                        "Ignoring gp compute ops of %s for %s since it is not supported on %s",
                        dtype,
                        op_invoke_info,
                        device_profile,
                    )
    compute_time_s = mma_ops_time_s + gp_ops_time_s
    memory_bandwidth = (
        device_profile.memory_bandwidth_bytes_ps * device_profile.memory_efficiency
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
            StatsKey.MEMORY_ACCESS: memory_access_time_s,
            StatsKey.COMPUTE: compute_time_s,
            StatsKey.MMA_OPS: mma_ops_time_s,
            StatsKey.GP_OPS: gp_ops_time_s,
            "is_compute_bound": compute_time_s > memory_access_time_s,
        },
    )
    return result


def _estimate_default(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
    result = _estimate_default_without_static_cost(op_invoke_info, device_profile)
    if result.execution_time_s == 0:
        return result
    result.execution_time_s += _estimate_static_cost(op_invoke_info, device_profile)
    return result


register_op_estimator(None, None)(_estimate_default)


@register_op_estimator(torch.ops.tensor_cast.all_reduce.default, None)
@register_op_estimator(torch.ops.tensor_cast.all_gather.default, None)
@register_op_estimator(torch.ops.tensor_cast.reduce_scatter.default, None)
@register_op_estimator(torch.ops.tensor_cast.all_to_all.default, None)
def _estimate_collective_comm(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
    from .comm_analytic import CommAnalyticModel

    result = _estimate_default_without_static_cost(op_invoke_info, device_profile)
    comm_model = CommAnalyticModel(device_profile)
    comm_result = comm_model.process_op(op_invoke_info)
    result.combine(comm_result)
    result.execution_time_s += device_profile.static_cost.comm_op_cost_s
    return result
