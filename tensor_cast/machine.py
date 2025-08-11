from typing import ClassVar, Dict, List
import torch
from dataclasses import dataclass, field


@dataclass
class MachineConfig:
    name: str

    DTYPES: ClassVar[List[torch.dtype]] = [torch.float32, torch.half, torch.bfloat16, torch.float8_e5m2, torch.int8]
    
    ops: Dict[torch.dtype, float] = field(default_factory=dict)
    memory_size_bytes: float = 0
    memory_bandwidth_bytes_ps: float = 0  # Bytes/s

    # TODO: add cache properties
    # TODO: add interconnnect properties

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

A2 = MachineConfig(
    name="910B",
    ops = {
        torch.float32: 44*1e12,
        torch.bfloat16: 280*1e12,
        torch.half: 280*1e12,
        torch.int8: 280*2*1e12,
    },
    memory_size_bytes=64*(1024**3),
    memory_bandwidth_bytes_ps=1.6*(1024**4),
)
