from typing import Dict
from overrides import override
from ..performance_model import PerformanceModel, OpInvokeInfo
from ..machine import MachineConfig


class EmpiricalPerformanceModel(PerformanceModel):
    """Performance model based on measured data"""
    def __init__(self, machine_config: MachineConfig):
        super().__init__("empirical", machine_config)

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        # TODO:
        return 0
