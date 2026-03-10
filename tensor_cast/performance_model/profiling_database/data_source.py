# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Abstract base class and data structures for the performance data source system.

This module defines the core interfaces that decouple EmpiricalPerformanceModel
from specific data backends (Profiling CSV, JIT benchmark cache, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..op_invoke_info import OpInvokeInfo


class QuerySource(Enum):
    """Indicates how a performance result was obtained."""

    MEASURED = auto()
    """Exact match from measured profiling data (confidence: 1.0)."""

    INTERPOLATED = auto()
    """Result estimated by interpolation between measured data points (confidence: 0.7-0.95)."""


@dataclass
class QueryResult:
    """Result returned by a DataSourcePerformanceModel lookup."""

    latency_us: float
    """Operator execution latency in microseconds."""

    confidence: float
    """Confidence score in [0, 1]. 1.0 means exact measured data."""

    source: QuerySource
    """How this result was obtained."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Optional extra metadata (e.g., matched CSV row info, interpolation weights)."""


class DataSourcePerformanceModel(ABC):
    """
    Abstract base class for operator performance data sources.

    TensorCast queries operator latency only through OpInvokeInfo; it is
    unaware of the underlying mapping logic, data format, or storage backend.
    """

    @abstractmethod
    def lookup(self, op_invoke_info: OpInvokeInfo) -> Optional[QueryResult]:
        """
        Query operator performance from the data source.
        """
        # TODO(Phase 1): implement interpolation
        ...

    def store(self, op_invoke_info: OpInvokeInfo, result: QueryResult) -> None:
        """
        Persist a performance measurement into this data source (optional).
        """
        # TODO(Phase 1): implement interpolation
        raise NotImplementedError(
            f"{type(self).__name__} is read-only and does not support store()"
        )
