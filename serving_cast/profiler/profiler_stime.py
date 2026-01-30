# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from stime import current_task_name, get_logger, now

logger = get_logger(__name__)

try:
    from ms_service_profiler import Profiler
except ImportError as e:
    raise ImportError("Please install ms_service_profiler") from e


class SimProfiler(Profiler):
    def event(self, event_name):
        self.metric("logical_start_time", now())
        self.metric("logical_end_time", now())
        self.metric("logical_pid", current_task_name())

        return super().event(event_name)

    def span_start(self, span_name):
        self.metric("logical_start_time", now())
        self.metric("logical_pid", current_task_name())

        return super().span_start(span_name)

    def span_end(self):
        self.metric("logical_end_time", now())
        self.metric("logical_pid", current_task_name())

        return super().span_end()
