import torch

from ..device import (
    CommGrid,
    DeviceProfile,
    InterconnectTopology,
    InterconnectType,
    StaticCost,
)


class ALIBABA:
    STATIC_COST = StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6, comm_op_cost_s=10 * 1e-6)

    # TODO(jgong5): double-confirm latency
    # TODO(jgong5): double-confirm communication efficiency
    INTERCONNECT = CommGrid(
        grid=torch.arange(32 * 4 * 4).reshape(32, 4, 4),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=50 * 1e9, latency_s=10 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=200 * 1e9, latency_s=1 * 1e-6, comm_efficiency=0.7
            ),
            2: InterconnectTopology(
                bandwidth_bytes_ps=200 * 1e9,
                latency_s=1 * 1e-6,
                comm_efficiency=0.7,
                type=InterconnectType.FULL_MESH,
            ),
        },
    )

    PPU = DeviceProfile(
        name="PPU",
        vendor="ALIBABA",
        mma_ops={
            torch.float32: 61 * 1e12,
            torch.bfloat16: 123 * 1e12,
            torch.half: 123 * 1e12,
            torch.int8: 236 * 1e12,
        },
        gp_ops={
            torch.float32: 25 * 1e12,
            torch.bfloat16: 25 * 1e12,
            torch.half: 25 * 1e12,
        },
        memory_size_bytes=96 * (1024**3),
        memory_bandwidth_bytes_ps=2.762 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT,
        static_cost=STATIC_COST,
    )
