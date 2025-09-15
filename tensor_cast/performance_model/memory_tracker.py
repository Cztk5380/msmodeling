import torch

from ..device import DeviceProfile
from ..performance_model import OpInvokeInfo


class MemoryTracker:
    """
    Tracks the memory allocation during the execution of a PyTorch program
    and gives its memory profiles.
    """

    def __init__(self, device_profile: DeviceProfile):
        self.device_profile = device_profile

    def track_tensor(tensor: torch.Tensor):
        # TODO
        pass

    def track_op_invocation(op_invoke_info: OpInvokeInfo):
        # TODO
        pass
