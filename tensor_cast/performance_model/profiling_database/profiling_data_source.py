# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from ...device import DeviceProfile  # noqa: TC001
from .data_source import DataSourcePerformanceModel, QueryResult

if TYPE_CHECKING:
    from ..op_invoke_info import OpInvokeInfo

logger = logging.getLogger(__name__)


class ProfilingDataSource(DataSourcePerformanceModel):
    """
    Directory layout expected under ``data_dir``::

        data_dir/
        ├── op_mapping.yaml          # operator name mapping + metadata
        ├── MatMulV2.csv             # one CSV per Profiling kernel Type
        ├── QuantBatchMatmulV3.csv
        └── ...
    Communication CSV files are stored in a separate directory referenced by
    ``communication_data_ref`` in ``op_mapping.yaml``::

        ../../hccl/v8.1.RC1/
        ├── comm_config.yaml
        ├── hcom_allReduce_.csv
        └── ...
    All CSV files preserve the original Profiling columns:
        Input Shapes, Input Data Types, Input Formats,
        Output Shapes, Output Data Types, Output Formats,
        Duration(us), ...
    Args:
        data_dir: Path to the versioned data directory containing
            ``op_mapping.yaml`` and the per-kernel-type CSV files.
        comm_grid: Optional CommGrid used to derive ``topology_tier`` for
            communication operator queries.
    """

    def __init__(self, data_dir: str, device_profile: DeviceProfile):
        """
        Initialize ProfilingDataSource.

        Should:
          1. Validate pandas / pyyaml availability.
          2. Load op_mapping.yaml from data_dir.
          3. Resolve communication_data_ref and load comm_config.yaml if present.
          4. Initialize an empty CSV cache (lazy loading).
        """
        raise NotImplementedError

    def lookup(self, op_invoke_info: OpInvokeInfo) -> Optional[QueryResult]:
        """
        Query operator performance from the Profiling CSV database.

        Dispatch path:
          1. Look up func name in op_mapping.yaml operator_mappings.
          2. If composite (e.g. MLA) -> return None (needs decomposition pass).
          3. If category == 'communication' -> _lookup_comm().
          4. If query_mode == 'attention_special' -> _lookup_attention().
          5. Default: _lookup_compute().

        Returns:
            QueryResult on hit, None on miss (caller should fall back to analytic model).
        """
        # TODO(Phase 1): implement interpolation
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"ProfilingDataSource(data_dir={self.data_dir})"
