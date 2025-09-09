from overrides import override

from ..device import DeviceProfile
from ..performance_model import OpInvokeInfo, PerformanceModel


class EmpiricalPerformanceModel(PerformanceModel):
    """Performance model based on measured data"""

    def __init__(self, machine_config: DeviceProfile):
        super().__init__("empirical", machine_config)

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        # TODO:
        return 0
