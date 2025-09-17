from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

import torch


@dataclass
class DeviceProfile:
    name: str

    all_machines: ClassVar[Dict[str, "DeviceProfile"]] = {}

    DTYPES: ClassVar[List[torch.dtype]] = [
        torch.float32,
        torch.half,
        torch.bfloat16,
        torch.float8_e5m2,
        torch.int8,
    ]

    mma_ops: Dict[torch.dtype, float] = field(default_factory=dict)
    gp_ops: Dict[torch.dtype, float] = field(default_factory=dict)
    compute_efficiency: float = 1.0
    memory_size_bytes: float = 0
    memory_bandwidth_bytes_ps: float = 0  # Bytes/s
    memory_efficiency: float = 1.0

    # TODO: add cache properties
    # TODO: add interconnnect properties

    def __post_init__(self):
        self.all_machines[self.name] = self


# TODO: name various SKUs correctly
# A3 = MachineConfig(
#     name="910C",
#     ops = {
#         torch.float32: 88*1e12,
#         torch.bfloat16: 560*1e12,
#         torch.half: 560*1e12,
#         torch.int8: 560*2*1e12,
#     },
#     memory_size_bytes=128*(1024**3),
#     memory_bandwidth_bytes_ps=1.6*2*(1024**4),
# )

A2 = DeviceProfile(
    name="A2",
    mma_ops={
        torch.float32: 140 * 1e12,
        torch.bfloat16: 280 * 1e12,
        torch.half: 280 * 1e12,
        torch.int8: 280 * 2 * 1e12,
    },
    gp_ops={
        torch.float32: 11 / 2 * 1e12,
        torch.bfloat16: 11 / 2 * 1e12,
        torch.half: 11 * 1e12,
    },
    memory_size_bytes=64 * (1024**3),
    memory_bandwidth_bytes_ps=1.6 * (1024**4),
    # The efficiencies are something we need to calibrate
    compute_efficiency=0.7,
    memory_efficiency=0.6,
)
