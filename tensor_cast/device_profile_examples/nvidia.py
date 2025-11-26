
import torch

from ..utils import DTYPE_FP4, DTYPE_FP8
from ..device import CommGrid, DeviceProfile, InterconnectTopology, StaticCost


class NVIDIA:
    # TODO(jgong5): double-confirm static cost
    STATIC_COST = StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6, comm_op_cost_s=10 * 1e-6)

    # TODO(jgong5): double-confirm latency
    # TODO(jgong5): double-confirm communication efficiency
    INTERCONNECT_PCIE_5 = CommGrid(
        grid=torch.arange(8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=63 * 1e9, latency_s=0.2 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    INTERCONNECT_NVLINK_IB = CommGrid(
        grid=torch.arange(128 * 8).reshape(128, 8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=50 * 1e9, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=450 * 1e9, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    INTERCONNECT_NVLINK_IB_16 = CommGrid(
        grid=torch.arange(128 * 16).reshape(128, 16),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=50 * 1e9, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=450 * 1e9, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    INTERCONNECT_RESTRICTED_NVLINK_IB = CommGrid(
        grid=torch.arange(128 * 8).reshape(128, 8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=50 * 1e9, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=200 * 1e9, latency_s=1.0 * 1e-6, comm_efficiency=0.7
            ),
        },
    )

    B30A = DeviceProfile(  # based on rumours
        name="B30A",
        vendor="NVIDIA",
        mma_ops={
            torch.float32: 750 * 1e12,
            torch.bfloat16: 1500 * 1e12,
            torch.half: 1500 * 1e12,
            DTYPE_FP8: 3000 * 1e12,
            torch.int8: 3000 * 1e12,
            DTYPE_FP4: 6000 * 1e12,
        },
        gp_ops={
            torch.float32: 120 * 1e12,
            torch.bfloat16: 120 * 1e12,
            torch.half: 120 * 1e12,
        },
        memory_size_bytes=144 * (1024**3),
        memory_bandwidth_bytes_ps=4.0 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT_NVLINK_IB_16,
        static_cost=STATIC_COST,
    )

    H20 = DeviceProfile(
        name="H20",
        vendor="NVIDIA",
        mma_ops={
            torch.float32: 74 * 1e12,
            torch.bfloat16: 148 * 1e12,
            torch.half: 148 * 1e12,
            DTYPE_FP8: 296 * 1e12,
            torch.int8: 296 * 1e12,
        },
        gp_ops={
            torch.float32: 44 * 1e12,
            torch.bfloat16: 44 * 1e12,
            torch.half: 44 * 1e12,
        },
        memory_size_bytes=96 * (1024**3),
        memory_bandwidth_bytes_ps=4.0 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT_NVLINK_IB,
        static_cost=STATIC_COST,
    )

    H100_SXM = DeviceProfile(
        name="H100_SXM",
        vendor="NVIDIA",
        mma_ops={
            torch.float32: 495 * 1e12,
            torch.bfloat16: 989.5 * 1e12,
            torch.half: 989.5 * 1e12,
            DTYPE_FP8: 1979 * 1e12,
            torch.int8: 1979 * 1e12,
        },
        gp_ops={
            torch.float32: 67 * 1e12,
            torch.bfloat16: 134 * 1e12,
            torch.half: 134 * 1e12,
        },
        memory_size_bytes=80 * (1024**3),
        memory_bandwidth_bytes_ps=3.35 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT_NVLINK_IB,
        static_cost=STATIC_COST,
    )

    H200_SXM = DeviceProfile(
        name="H200_SXM",
        vendor="NVIDIA",
        mma_ops={
            torch.float32: 495 * 1e12,
            torch.bfloat16: 989.5 * 1e12,
            torch.half: 989.5 * 1e12,
            DTYPE_FP8: 1979 * 1e12,
            torch.int8: 1979 * 1e12,
        },
        gp_ops={
            torch.float32: 67 * 1e12,
            torch.bfloat16: 134 * 1e12,
            torch.half: 134 * 1e12,
        },
        memory_size_bytes=141 * (1024**3),
        memory_bandwidth_bytes_ps=4.8 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT_NVLINK_IB,
        static_cost=STATIC_COST,
    )

    H800_SXM = DeviceProfile(
        name="H800_SXM",
        vendor="NVIDIA",
        mma_ops={
            torch.float32: 495 * 1e12,
            torch.bfloat16: 989.5 * 1e12,
            torch.half: 989.5 * 1e12,
            DTYPE_FP8: 1979 * 1e12,
            torch.int8: 1979 * 1e12,
        },
        gp_ops={
            torch.float32: 67 * 1e12,
            torch.bfloat16: 134 * 1e12,
            torch.half: 134 * 1e12,
        },
        memory_size_bytes=80 * (1024**3),
        memory_bandwidth_bytes_ps=3.35 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT_RESTRICTED_NVLINK_IB,
        static_cost=STATIC_COST,
    )

    L20 = DeviceProfile(
        name="L20",
        vendor="NVIDIA",
        mma_ops={
            torch.float32: 59.8 * 1e12,
            torch.bfloat16: 119.5 * 1e12,
            torch.half: 119.5 * 1e12,
            DTYPE_FP8: 239 * 1e12,
            torch.int8: 239 * 1e12,
        },
        gp_ops={
            torch.float32: 59.8 * 1e12,
            torch.bfloat16: 59.8 * 1e12,
            torch.half: 59.8 * 1e12,
        },
        memory_size_bytes=48 * (1024**3),
        memory_bandwidth_bytes_ps=0.864 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT_PCIE_5,
        static_cost=STATIC_COST,
    )

    RTX_PRO_6000D = DeviceProfile(
        name="RTX_PRO_6000D",
        vendor="NVIDIA",
        mma_ops={
            torch.float32: 74 * 1e12,
            torch.bfloat16: 148 * 1e12,
            torch.half: 148 * 1e12,
            DTYPE_FP8: 296 * 1e12,
            torch.int8: 296 * 1e12,
        },
        gp_ops={
            torch.float32: 75 * 1e12,
            torch.bfloat16: 75 * 1e12,
            torch.half: 75 * 1e12,
        },
        memory_size_bytes=96 * (1024**3),
        memory_bandwidth_bytes_ps=1.4 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT_PCIE_5,
        static_cost=STATIC_COST,
    )

    RTX_6000D = DeviceProfile(
        name="RTX_6000D",
        vendor="NVIDIA",
        mma_ops={
            torch.float32: 74 * 1e12,
            torch.bfloat16: 148 * 1e12,
            torch.half: 148 * 1e12,
            DTYPE_FP8: 296 * 1e12,
            torch.int8: 296 * 1e12,
        },
        gp_ops={
            torch.float32: 91.1 * 1e12,
            torch.bfloat16: 91.1 * 1e12,
            torch.half: 91.1 * 1e12,
        },
        memory_size_bytes=96 * (1024**3),
        memory_bandwidth_bytes_ps=1.4 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT_PCIE_5,
        static_cost=STATIC_COST,
    )

    RTX_5090D = DeviceProfile(
        name="RTX_5090D",
        vendor="NVIDIA",
        mma_ops={
            torch.float32: 104.8 * 1e12,
            torch.bfloat16: 297 * 1e12,
            torch.half: 297 * 1e12,
            DTYPE_FP8: 593 * 1e12,
            torch.int8: 593 * 1e12,
            DTYPE_FP4: 1186 * 1e12,
        },
        gp_ops={
            torch.float32: 104.8 * 1e12,
            torch.bfloat16: 104.8 * 1e12,
            torch.half: 104.8 * 1e12,
        },
        memory_size_bytes=32 * (1024**3),
        memory_bandwidth_bytes_ps=1.792 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT_PCIE_5,
        static_cost=STATIC_COST,
    )

    RTX_5090Dv2 = DeviceProfile(
        name="RTX_5090Dv2",
        vendor="NVIDIA",
        mma_ops={
            torch.float32: 104.8 * 1e12,
            torch.bfloat16: 297 * 1e12,
            torch.half: 297 * 1e12,
            DTYPE_FP8: 593 * 1e12,
            torch.int8: 593 * 1e12,
            DTYPE_FP4: 1186 * 1e12,
        },
        gp_ops={
            torch.float32: 104.8 * 1e12,
            torch.bfloat16: 104.8 * 1e12,
            torch.half: 104.8 * 1e12,
        },
        memory_size_bytes=24 * (1024**3),
        memory_bandwidth_bytes_ps=1.34 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT_PCIE_5,
        static_cost=STATIC_COST,
    )

    RTX_4090 = DeviceProfile(
        name="RTX_4090",
        vendor="NVIDIA",
        mma_ops={
            torch.float32: 115 * 1e12,
            torch.bfloat16: 330.3 * 1e12,
            torch.half: 330.3 * 1e12,
            torch.int8: 660.6 * 1e12,
        },
        gp_ops={
            torch.float32: 82.6 * 1e12,
            torch.bfloat16: 82.6 * 1e12,
            torch.half: 82.6 * 1e12,
        },
        memory_size_bytes=24 * (1024**3),
        memory_bandwidth_bytes_ps=1.008 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT_PCIE_5,
        static_cost=STATIC_COST,
    )

    RTX_4090D = DeviceProfile(
        name="RTX_4090D",
        vendor="NVIDIA",
        mma_ops={
            torch.float32: 73.5 * 1e12,
            torch.bfloat16: 220 * 1e12,
            torch.half: 294 * 1e12,
            torch.int8: 588 * 1e12,
        },
        gp_ops={
            torch.float32: 73.5 * 1e12,
            torch.bfloat16: 73.5 * 1e12,
            torch.half: 73.5 * 1e12,
        },
        memory_size_bytes=24 * (1024**3),
        memory_bandwidth_bytes_ps=1.008 * (1024**4),
        # The efficiencies are something we need to calibrate
        compute_efficiency=0.7,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT_PCIE_5,
        static_cost=STATIC_COST,
    )


