from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

import torch


@dataclass
class InterconnectTopology:
    # TODO(jgong5): support specifying various topology types like AllToAll, Torus, FatTree etc.
    bandwidth_bytes_ps: float  # unidirectional bandwidth (GB/s)
    latency_s: float
    comm_efficiency: float = 1.0


@dataclass
class CommGrid:
    """A communication grid of devices and how they are interconnected"""

    grid: torch.Tensor
    """
    An hierarchical interconnect structure of devices usually faster with inner dims
    and slower with outer dims. For example,
    A grid with 256 devices could be arranged in [16, 8, 2] where the inner-most dim "2"
    representing the fastest MCP connecting two devices and the middle dim "8" groups 8
    such 2-device packaging in a server "node" and the outer-most dim "16" groups 16 of
    the server nodes.
    """

    topologies: Dict[int, InterconnectTopology]
    """
    Map start_dim in the grid to an interconnect topology.

    The mapping of the device grid to the interconnected topologies. Basically, it maps a single
    or multiple dims of device grids to some topology. Note that a particular dim of the grid
    can be mapped to multiple topologies. For example, a grid of 256 devices mentioned previously
    can have the inner-most dim "2" mapped to "AllToAll", the inner-most two dims [8, 2] can be
    mapped to "AllToAll" with a bit slower connection and then all the devices [16, 8, 2] are mapped
    to a slowest "FatTree" interconnect.
    """


@dataclass
class DeviceProfile:
    name: str
    comm_grid: CommGrid

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

A2_256_commgrid = CommGrid(
    grid=torch.arange(256 * 8).reshape(256, 8),
    topologies={
        0: InterconnectTopology(
            bandwidth_bytes_ps=50, latency_s=1e-5, comm_efficiency=0.7
        ),
        1: InterconnectTopology(
            bandwidth_bytes_ps=196, latency_s=1.3e-6, comm_efficiency=0.7
        ),
    },
)

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
    comm_grid=A2_256_commgrid,
)
