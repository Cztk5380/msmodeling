# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
profiling_database: operator performance data source system for TensorCast.

Public API (defined by Liuren team):
    DataSourcePerformanceModel               - abstract base class for all data sources
    QueryResult                              - dataclass returned by DataSourcePerformanceModel.lookup()
    QuerySource                              - enum indicating how a result was obtained
    ProfilingDataSource                      - read-only DataSourcePerformanceModel backed by Profiling CSV files
                                               (skeleton - implementation by Xiaoqiaoling team)
    InterpolatingDataSource                  - wrapper DataSourcePerformanceModel that adds interpolation capability
"""

from .data_source import DataSourcePerformanceModel, QueryResult, QuerySource
from .interpolating_data_source import InterpolatingDataSource
from .profiling_data_source import ProfilingDataSource

__all__ = [
    "DataSourcePerformanceModel",
    "QueryResult",
    "QuerySource",
    "ProfilingDataSource",
    "InterpolatingDataSource",
]
