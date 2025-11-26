import torch

from ..device import (
    CommGrid,
    DeviceProfile,
    InterconnectTopology,
    InterconnectType,
    StaticCost,
)


class METAX:
    STATIC_COST = StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6, comm_op_cost_s=10 * 1e-6)

    # TODO(jgong5): double-confirm latency
    # TODO(jgong5): double-confirm communication efficiency
    INTERCONNECT = CommGrid(
        grid=torch.arange(64 * 8).reshape(64, 8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=50 * 1e9, latency_s=10 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=224 * 1e9,
                latency_s=1 * 1e-6,
                comm_efficiency=0.7,
                type=InterconnectType.FULL_MESH,
            ),
        },
    )

    BW1000 = DeviceProfile(
        name="BW1000",
        vendor="HYGON",
        mma_ops={
            torch.float32: 78 * 1e12,
            torch.bfloat16: 420 * 1e12,
            torch.half: 840 * 1e12,
            torch.int8: 982 * 1e12,
        },
        gp_ops={
            torch.float32: 26 * 1e12,
            torch.bfloat16: 26 * 1e12,
            torch.half: 26 * 1e12,
        },
        memory_size_bytes=64 * (1024**3),
        memory_bandwidth_bytes_ps=1.6 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT,
        static_cost=STATIC_COST,
    )
