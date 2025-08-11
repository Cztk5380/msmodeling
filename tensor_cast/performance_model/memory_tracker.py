import torch
from ..performance_model import OpInvokeInfo
from ..machine import MachineConfig

class MemoryTracker:
    """
    Tracks the memory allocation during the execution of a PyTorch program
    and gives its memory profiles.
    """
    def __init__(self, machine_config: MachineConfig):
        self.machine_config = machine_config

    def track_tensor(tensor: torch.Tensor):
        # TODO
        pass

    def track_op_invocation(op_invoke_info: OpInvokeInfo):
        # TODO
        pass