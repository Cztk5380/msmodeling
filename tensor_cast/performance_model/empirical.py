# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List, Optional

from overrides import override

from ..device import DeviceProfile
from .analytic import AnalyticPerformanceModel
from .base import PerformanceModel
from .op_invoke_info import OpInvokeInfo
from .profiling_database.data_source import DataSourcePerformanceModel


class EmpiricalPerformanceModel(PerformanceModel):
    """
    Performance model that queries operator latency from a DataSourcePerformanceModel.

    On a cache hit the latency from the DataSourcePerformanceModel is returned directly.
    On a miss, the call is forwarded to ``fallback_model`` (default:
    AnalyticPerformanceModel / Roofline).

    Usage examples::

        # Backed by pre-collected Profiling CSV files
        from tensor_cast.performance_model.profiling_database import ProfilingDataSource

        data_source = ProfilingDataSource(
            "profiling_database/data/atlas_a3_752t_128g/vllm_ascend/v0.13.0",
            comm_grid=device_profile.comm_grid,
        )

        pm = EmpiricalPerformanceModel(
            device_profile,
            data_source=InterpolatingDataSource(
                ProfilingDataSource(
                    "profiling_database/data/atlas_a3_752t_128g/vllm_ascend/v0.13.0"
                )
            ),
        )

    Args:
        device_profile: Hardware device profile.
        data_source: DataSourcePerformanceModel implementation to query latency from.
        fallback_model: Performance model used when the data source returns None.
            Defaults to AnalyticPerformanceModel (Roofline model).
    """

    def __init__(
        self,
        device_profile: DeviceProfile,
        data_source: DataSourcePerformanceModel,
        fallback_model: Optional[PerformanceModel] = None,
    ):
        super().__init__("empirical", device_profile)
        self.data_source = data_source
        self.fallback_model = fallback_model or AnalyticPerformanceModel(device_profile)

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        result = self.data_source.lookup(op_invoke_info)
        if result is not None:
            return PerformanceModel.Result(
                execution_time_s=result.latency_us * 1e-6,
                statistics={
                    "source": result.source.name,
                    "confidence": result.confidence,
                },
            )
        # Data source miss: delegate to fallback model (Roofline / CommAnalytic)
        return self.fallback_model.process_op(op_invoke_info)

    @override
    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        """
        Return classifiers from the fallback model so that breakdown reporting
        still works when an op is handled by the fallback path.
        """
        return self.fallback_model.get_classifiers()
