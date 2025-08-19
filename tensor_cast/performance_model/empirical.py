from overrides import override

from ..machine import MachineConfig
from ..performance_model import OpInvokeInfo, PerformanceModel


class EmpiricalPerformanceModel(PerformanceModel):
    """Performance model based on measured data"""

    def __init__(self, machine_config: MachineConfig):
        super().__init__("empirical", machine_config)

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        # TODO:
        return 0
