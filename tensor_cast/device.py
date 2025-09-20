from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

import torch


DTYPE_FP4 = torch.int4  # use int4 placeholder for FP4


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
class StaticCost:
    """Device-side scheduling cost of individual ops"""

    mma_op_cost_s: float = 0
    gp_op_cost_s: float = 0
    comm_op_cost_s: float = 0


@dataclass
class DeviceProfile:
    name: str
    comm_grid: CommGrid

    all_device_profiles: ClassVar[Dict[str, "DeviceProfile"]] = {}

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

    static_cost: StaticCost = field(default_factory=StaticCost)

    # TODO: add cache properties

    def __post_init__(self):
        if self.name in self.all_device_profiles:
            raise ValueError(f"{self.name} already exists")
        self.all_device_profiles[self.name] = self


TEST_INTERCONNECT = CommGrid(
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

TEST_DEVICE = DeviceProfile(
    name="TEST_DEVICE",
    mma_ops={
        torch.float32: 99.5 * 1e12,
        torch.bfloat16: 353.9 * 1e12,
        torch.half: 353.9 * 1e12,
        torch.int8: 353.9 * 2 * 1e12,
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
    comm_grid=TEST_INTERCONNECT,
    static_cost=StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6),
)


class NVIDIA:
    # TODO(jgong5): double-confirm static cost
    STATIC_COST = StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6)

    # TODO(jgong5): double-confirm latency
    # TODO(jgong5): double-confirm communication efficiency
    INTERCONNECT_PCIE_5 = CommGrid(
        grid=torch.arange(8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=63, latency_s=0.2 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    INTERCONNECT_NVLINK_IB = CommGrid(
        grid=torch.arange(128 * 8).reshape(128, 8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=50, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=450, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    INTERCONNECT_RESTRICTED_NVLINK_IB = CommGrid(
        grid=torch.arange(128 * 8).reshape(128, 8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=50, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=200, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    B30A = DeviceProfile(  # based on rumours
        name="B30A",
        mma_ops={
            torch.float32: 41.5 * 1e12,
            torch.bfloat16: 937.5 * 1e12,
            torch.half: 937.5 * 1e12,
            torch.float8_e5m2: 1875 * 1e12,
            torch.int8: 1875 * 1e12,
            DTYPE_FP4: 3750 * 1e12,
        },
        gp_ops={
            torch.float32: 41.5 * 1e12,
            torch.bfloat16: 41.5 * 1e12,
            torch.half: 83 * 1e12,
        },
        memory_size_bytes=144 * (1024**3),
        memory_bandwidth_bytes_ps=4.0 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT_NVLINK_IB,
        static_cost=STATIC_COST,
    )

    H20 = DeviceProfile(
        name="H20",
        mma_ops={
            torch.float32: 44 * 1e12,
            torch.bfloat16: 148 * 1e12,
            torch.half: 148 * 1e12,
            torch.float8_e5m2: 296 * 1e12,
            torch.int8: 296 * 1e12,
        },
        gp_ops={
            torch.float32: 44 * 1e12,
            torch.bfloat16: 44 * 1e12,
            torch.half: 88 * 1e12,
        },
        memory_size_bytes=96 * (1024**3),
        memory_bandwidth_bytes_ps=4.0 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT_NVLINK_IB,
        static_cost=STATIC_COST,
    )

    H100_SXM = DeviceProfile(
        name="H100_SXM",
        mma_ops={
            torch.float32: 67 * 1e12,
            torch.bfloat16: 989.5 * 1e12,
            torch.half: 989.5 * 1e12,
            torch.float8_e5m2: 1979 * 1e12,
            torch.int8: 1979 * 1e12,
        },
        gp_ops={
            torch.float32: 67 * 1e12,
            torch.bfloat16: 67 * 1e12,
            torch.half: 134 * 1e12,
        },
        memory_size_bytes=80 * (1024**3),
        memory_bandwidth_bytes_ps=3.35 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT_NVLINK_IB,
        static_cost=STATIC_COST,
    )

    H200_SXM = DeviceProfile(
        name="H200_SXM",
        mma_ops={
            torch.float32: 67 * 1e12,
            torch.bfloat16: 989.5 * 1e12,
            torch.half: 989.5 * 1e12,
            torch.float8_e5m2: 1979 * 1e12,
            torch.int8: 1979 * 1e12,
        },
        gp_ops={
            torch.float32: 67 * 1e12,
            torch.bfloat16: 67 * 1e12,
            torch.half: 134 * 1e12,
        },
        memory_size_bytes=141 * (1024**3),
        memory_bandwidth_bytes_ps=4.8 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT_NVLINK_IB,
        static_cost=STATIC_COST,
    )

    H800_SXM = DeviceProfile(
        name="H800_SXM",
        mma_ops={
            torch.float32: 67 * 1e12,
            torch.bfloat16: 989.5 * 1e12,
            torch.half: 989.5 * 1e12,
            torch.float8_e5m2: 1979 * 1e12,
            torch.int8: 1979 * 1e12,
        },
        gp_ops={
            torch.float32: 67 * 1e12,
            torch.bfloat16: 67 * 1e12,
            torch.half: 134 * 1e12,
        },
        memory_size_bytes=80 * (1024**3),
        memory_bandwidth_bytes_ps=3.35 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT_RESTRICTED_NVLINK_IB,
        static_cost=STATIC_COST,
    )

    L20 = DeviceProfile(
        name="L20",
        mma_ops={
            torch.float32: 59.8 * 1e12,
            torch.bfloat16: 119.5 * 1e12,
            torch.half: 119.5 * 1e12,
            torch.float8_e5m2: 239 * 1e12,
            torch.int8: 239 * 1e12,
        },
        gp_ops={
            torch.float32: 59.8 * 1e12,
            torch.bfloat16: 59.8 * 1e12,
            torch.half: 119.6 * 1e12,
        },
        memory_size_bytes=48 * (1024**3),
        memory_bandwidth_bytes_ps=0.864 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT_PCIE_5,
        static_cost=STATIC_COST,
    )

    RTX_6000D = DeviceProfile(
        name="RTX_6000D",
        mma_ops={
            torch.float32: 91.1 * 1e12,
            torch.bfloat16: 148 * 1e12,
            torch.half: 148 * 1e12,
            torch.float8_e5m2: 296 * 1e12,
            torch.int8: 296 * 1e12,
        },
        gp_ops={
            torch.float32: 91.1 * 1e12,
            torch.bfloat16: 91.1 * 1e12,
            torch.half: 182.2 * 1e12,
        },
        memory_size_bytes=96 * (1024**3),
        memory_bandwidth_bytes_ps=1.4 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT_PCIE_5,
        static_cost=STATIC_COST,
    )

    RTX_5090D = DeviceProfile(
        name="RTX_5090D",
        mma_ops={
            torch.float32: 104.8 * 1e12,
            torch.bfloat16: 297 * 1e12,
            torch.half: 297 * 1e12,
            torch.float8_e5m2: 593 * 1e12,
            torch.int8: 593 * 1e12,
            DTYPE_FP4: 1186 * 1e12,
        },
        gp_ops={
            torch.float32: 104.8 * 1e12,
            torch.bfloat16: 104.8 * 1e12,
            torch.half: 209.6 * 1e12,
        },
        memory_size_bytes=32 * (1024**3),
        memory_bandwidth_bytes_ps=1.792 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT_PCIE_5,
        static_cost=STATIC_COST,
    )

    RTX_4090 = DeviceProfile(
        name="RTX_4090",
        mma_ops={
            torch.float32: 82.6 * 1e12,
            torch.bfloat16: 247.8 * 1e12,
            torch.half: 330.3 * 1e12,
            torch.int8: 660.6 * 1e12,
        },
        gp_ops={
            torch.float32: 82.6 * 1e12,
            torch.bfloat16: 82.6 * 1e12,
            torch.half: 165.2 * 1e12,
        },
        memory_size_bytes=24 * (1024**3),
        memory_bandwidth_bytes_ps=1.008 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT_PCIE_5,
        static_cost=STATIC_COST,
    )

    RTX_4090D = DeviceProfile(
        name="RTX_4090D",
        mma_ops={
            torch.float32: 73.5 * 1e12,
            torch.bfloat16: 220 * 1e12,
            torch.half: 294 * 1e12,
            torch.int8: 588 * 1e12,
        },
        gp_ops={
            torch.float32: 73.5 * 1e12,
            torch.bfloat16: 73.5 * 1e12,
            torch.half: 147 * 1e12,
        },
        memory_size_bytes=24 * (1024**3),
        memory_bandwidth_bytes_ps=1.008 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT_PCIE_5,
        static_cost=STATIC_COST,
    )


class CAMBRICON:
    # TODO(jgong5): double-confirm static cost
    STATIC_COST = StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6)

    # TODO(jgong5): double-confirm latency
    # TODO(jgong5): double-confirm communication efficiency
    INTERCONNECT = CommGrid(
        grid=torch.arange(128 * 8).reshape(128, 8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=25, latency_s=10.0 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=293, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    MLU590 = DeviceProfile(
        name="MLU590",
        mma_ops={
            torch.float32: 104.8 * 1e12,
            torch.bfloat16: 297 * 1e12,
            torch.half: 297 * 1e12,
            torch.float8_e5m2: 593 * 1e12,
            torch.int8: 593 * 1e12,
            DTYPE_FP4: 1186 * 1e12,
        },
        gp_ops={
            torch.float32: 26 * 1e12,
            torch.bfloat16: 54 * 1e12,
            torch.half: 54 * 1e12,
        },
        memory_size_bytes=32 * (1024**3),
        memory_bandwidth_bytes_ps=1.792 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT,
        static_cost=STATIC_COST,
    )


class KUNLUNXIN:
    STATIC_COST = StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6)

    # TODO(jgong5): double-confirm latency
    # TODO(jgong5): double-confirm communication efficiency
    INTERCONNECT = CommGrid(
        grid=torch.arange(16 * 4).reshape(16, 4),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=200, latency_s=20 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=200, latency_s=10 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    P800 = DeviceProfile(
        name="P800",
        mma_ops={
            torch.float32: 87.5 * 1e12,
            torch.bfloat16: 350 * 1e12,
            torch.half: 350 * 1e12,
            torch.int8: 700 * 1e12,
        },
        gp_ops={
            torch.float32: 87.5 * 1e12,
            torch.bfloat16: 87.5 * 1e12,
            torch.half: 175 * 1e12,
        },
        memory_size_bytes=96 * (1024**3),
        memory_bandwidth_bytes_ps=2.7 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT,
        static_cost=STATIC_COST,
    )


class ALIBABA:
    STATIC_COST = StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6)

    # TODO(jgong5): double-confirm latency
    # TODO(jgong5): double-confirm communication efficiency
    INTERCONNECT = CommGrid(
        grid=torch.arange(4 * 16).reshape(4, 16),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=50, latency_s=10 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=150, latency_s=1 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    PPU = DeviceProfile(
        name="PPU",
        mma_ops={
            torch.float32: 25 * 1e12,
            torch.bfloat16: 123 * 1e12,
            torch.half: 123 * 1e12,
            torch.int8: 236 * 1e12,
        },
        gp_ops={
            torch.float32: 25 * 1e12,
            torch.bfloat16: 25 * 1e12,
            torch.half: 50 * 1e12,
        },
        memory_size_bytes=96 * (1024**3),
        memory_bandwidth_bytes_ps=2.762 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT,
        static_cost=STATIC_COST,
    )


class METAX:
    STATIC_COST = StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6)

    # TODO(jgong5): double-confirm latency
    # TODO(jgong5): double-confirm communication efficiency
    INTERCONNECT = CommGrid(
        grid=torch.arange(8 * 8).reshape(8, 8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=12.5, latency_s=10 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=448, latency_s=1 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    C550 = DeviceProfile(
        name="C550",
        mma_ops={
            torch.float32: 54 * 1e12,
            torch.bfloat16: 280 * 1e12,
            torch.half: 280 * 1e12,
            torch.int8: 560 * 1e12,
        },
        gp_ops={
            torch.float32: 54 * 1e12,
            torch.bfloat16: 54 * 1e12,
            torch.half: 108 * 1e12,
        },
        memory_size_bytes=64 * (1024**3),
        memory_bandwidth_bytes_ps=1.6 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT,
        static_cost=STATIC_COST,
    )
