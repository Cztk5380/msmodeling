import torch

from ..device import (
    CommGrid,
    DeviceProfile,
    InterconnectTopology,
    InterconnectType,
    StaticCost,
)


class KUNLUNXIN:
    STATIC_COST = StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6)

    # TODO(jgong5): double-confirm latency
    # TODO(jgong5): double-confirm communication efficiency
    INTERCONNECT = CommGrid(
        grid=torch.arange(16 * 4).reshape(16, 4),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=200 * 1e9, latency_s=20 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=200 * 1e9,
                latency_s=10 * 1e-6,
                comm_efficiency=0.7,
                type=InterconnectType.FULL_MESH,
            ),
        },
    )

    P800 = DeviceProfile(
        name="P800",
        vendor="BAIDU",
        mma_ops={
            torch.float32: 175 * 1e12,
            torch.bfloat16: 350 * 1e12,
            torch.half: 350 * 1e12,
            torch.int8: 700 * 1e12,
        },
        gp_ops={
            torch.float32: 40 * 1e12,
            torch.bfloat16: 40 * 1e12,
            torch.half: 40 * 1e12,
        },
        memory_size_bytes=96 * (1024**3),
        memory_bandwidth_bytes_ps=2.7 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.6,
        comm_grid=INTERCONNECT,
        static_cost=STATIC_COST,
    )
