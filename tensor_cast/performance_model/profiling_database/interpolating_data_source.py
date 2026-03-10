# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from .data_source import DataSourcePerformanceModel, QueryResult, QuerySource

if TYPE_CHECKING:
    from ..op_invoke_info import OpInvokeInfo

logger = logging.getLogger(__name__)


class InterpolatingDataSource(DataSourcePerformanceModel):
    """
    Wrapper DataSourcePerformanceModel that adds interpolation / approximate query capability
    on top of an underlying base DataSourcePerformanceModel.

    Phase 1 behavior: delegates all queries to base_source only (exact match).
    Phase 2 will implement:
      - 2D+1D mixed interpolation (reference: AI Configurator)
      - sqrt transform on seq_len dimension for O(n^2) attention operators
      - Confidence scoring based on distance to nearest measured data points

    Args:
        base_source: The underlying DataSourcePerformanceModel to query first.
        interpolation_config: Optional interpolation policy dict loaded from
            op_mapping.yaml's 'interpolation_policy' section.
    """

    def __init__(
        self,
        base_source: DataSourcePerformanceModel,
        interpolation_config: Optional[dict] = None,
    ):
        self.base_source = base_source
        self.interpolation_config = interpolation_config

    def lookup(self, op_invoke_info: OpInvokeInfo) -> Optional[QueryResult]:
        """
        Query operator performance.

        Phase 1: exact match only, delegates to base_source.
        Phase 2: will attempt interpolation if no exact match is found.

        Args:
            op_invoke_info: Metadata describing the operator invocation.

        Returns:
            QueryResult on hit, None if no data available.
        """
        # Step 1: try exact match from the base data source
        result = self.base_source.lookup(op_invoke_info)
        if result is not None and result.source == QuerySource.MEASURED:
            return result

        # TODO(Phase 2): implement neighbor search and interpolation
        # Step 2: find neighboring measured data points
        # neighbors = self._find_neighbors(op_invoke_info)
        # if not neighbors:
        #     return None
        # Step 3: interpolate / extrapolate
        # return self._interpolate(neighbors, op_invoke_info)

        return result  # return whatever base_source returned (could be None)

    def _find_neighbors(self, op_invoke_info: OpInvokeInfo):
        """
        Find neighboring measured data points for interpolation.

        Returns:
            List of (QueryResult, distance) tuples, or empty list if none found.
        """
        # TODO(Phase 2): implement neighbor search
        raise NotImplementedError("Neighbor search will be implemented in Phase 2")

    def _interpolate(
        self, neighbors, op_invoke_info: OpInvokeInfo
    ) -> Optional[QueryResult]:
        """
        Interpolate latency from neighboring measured data points.

        Returns:
            QueryResult with interpolated latency and confidence, or None.
        """
        # TODO(Phase 2): implement interpolation
        raise NotImplementedError("Interpolation will be implemented in Phase 2")

    def __repr__(self) -> str:
        return f"InterpolatingDataSource(base_source={self.base_source!r})"
