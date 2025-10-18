import torch

from ..utils import DTYPE_FP4, DTYPE_FP8
from ..device import CommGrid, DeviceProfile, InterconnectTopology, StaticCost



class CAMBRICON:
    # TODO(jgong5): double-confirm static cost
    STATIC_COST = StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6)

    # TODO(jgong5): double-confirm latency
    # TODO(jgong5): double-confirm communication efficiency
    INTERCONNECT = CommGrid(
        grid=torch.arange(128 * 8).reshape(128, 8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=25 * 1e9, latency_s=10.0 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=200 * 1e9, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    INTERCONNECT_690 = CommGrid(
        grid=torch.arange(128 * 8).reshape(128, 8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=25 * 1e9, latency_s=10.0 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=400 * 1e9, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    INTERCONNECT_PCIE = CommGrid(
        grid=torch.arange(8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=32 * 1e9, latency_s=10.0 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    MLU690 = DeviceProfile(
        name="MLU690",
        vendor="CAMBRICON",
        mma_ops={
            torch.float32: 406 * 1e12,
            torch.bfloat16: 813 * 1e12,
            torch.half: 813 * 1e12,
            DTYPE_FP8: 1626 * 1e12,
            torch.int8: 1626 * 1e12,
            DTYPE_FP4: 3252 * 1e12,
        },
        gp_ops={
            torch.float32: 51 * 1e12,
            torch.bfloat16: 51 * 1e12,
            torch.half: 51 * 1e12,
        },
        memory_size_bytes=192 * (1024**3),
        memory_bandwidth_bytes_ps=8 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT_690,
        static_cost=STATIC_COST,
    )

    MLU590 = DeviceProfile(
        name="MLU590",
        vendor="CAMBRICON",
        mma_ops={
            torch.float32: 157 * 1e12,
            torch.bfloat16: 314 * 1e12,
            torch.half: 314 * 1e12,
            DTYPE_FP8: 628 * 1e12,
            torch.int8: 628 * 1e12,
            DTYPE_FP4: 1256 * 1e12,
        },
        gp_ops={
            torch.float32: 20 * 1e12,
            torch.bfloat16: 20 * 1e12,
            torch.half: 20 * 1e12,
        },
        memory_size_bytes=96 * (1024**3),
        memory_bandwidth_bytes_ps=2.7 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT,
        static_cost=STATIC_COST,
    )

    MLU580 = DeviceProfile(
        name="MLU580",
        vendor="CAMBRICON",
        mma_ops={
            torch.float32: 157 * 1e12,
            torch.bfloat16: 275 * 1e12,
            torch.half: 275 * 1e12,
            DTYPE_FP8: 550 * 1e12,
            torch.int8: 550 * 1e12,
        },
        gp_ops={
            torch.float32: 17 * 1e12,
            torch.bfloat16: 17 * 1e12,
            torch.half: 17 * 1e12,
        },
        memory_size_bytes=48 * (1024**3),
        memory_bandwidth_bytes_ps=1.2 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT_PCIE,
        static_cost=STATIC_COST,
    )


