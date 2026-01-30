from overrides import override

from ..device import DeviceProfile
from .base import PerformanceModel
from .op_benchmark import OpBenchmark
from .op_invoke_info import OpInvokeInfo


class EmpiricalPerformanceModel(PerformanceModel):
    """Performance model based on measured data"""

    def __init__(self, device_profile: DeviceProfile):
        # TODO(jgong5): add a mode so that we can do JIT or offline benchmarks
        super().__init__("empirical", device_profile)
        self.op_benchmark = OpBenchmark(device_profile)

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        return self.op_benchmark.benchmark(op_invoke_info)
