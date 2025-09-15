from overrides import override

from ..device import DeviceProfile
from ..performance_model import OpInvokeInfo, PerformanceModel


class EmpiricalPerformanceModel(PerformanceModel):
    """Performance model based on measured data"""

    def __init__(self, device_profile: DeviceProfile):
        super().__init__("empirical", device_profile)

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        # TODO:
        return 0
