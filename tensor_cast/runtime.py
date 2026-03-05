import collections
import contextlib
import dataclasses
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Union

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .device import DeviceProfile
from .patch_torch import patch_torch
from .performance_model.base import CachingPerformanceModel, PerformanceModel
from .performance_model.memory_tracker import MemoryTracker
from .performance_model.op_invoke_info import OpInvokeInfo, Region

logger = logging.getLogger(__name__)

_current_runtime = threading.local()


def current_runtime():
    return getattr(_current_runtime, "value", None)


@dataclasses.dataclass
class RuntimeEvent:
    op_invoke_info: OpInvokeInfo
    perf_results: Dict[str, PerformanceModel.Result] = dataclasses.field(
        default_factory=dict
    )


class Runtime(TorchDispatchMode):
    """
    Runtime of TensorCast that simulates the execution of a PyTorch program.
    """

    def __init__(
        self,
        perf_models: Union[PerformanceModel, List[PerformanceModel]],
        device_profile: DeviceProfile,
        memory_tracker: Optional[MemoryTracker] = None,
    ):
        super().__init__()
        self.perf_models = (
            perf_models if isinstance(perf_models, (list, tuple)) else [perf_models]
        )
        self.perf_models = [
            perf_model
            if isinstance(perf_model, CachingPerformanceModel)
            else CachingPerformanceModel(perf_model)
            for perf_model in self.perf_models
        ]
        self.device_profile = device_profile
        self.memory_tracker: Optional[MemoryTracker] = memory_tracker
        self.op_invoke_infos: List[OpInvokeInfo] = []
        self.op_info_group: List[Union[OpInvokeInfo, Region]] = []
        self.event_list: List[RuntimeEvent] = []
        # TODO: add multi-stream support

        self.exit_stack = contextlib.ExitStack()

    @classmethod
    def is_infra_mode(cls):
        return True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if not torch.compiler.is_compiling():
            func_name = (
                func.__qualname__ if hasattr(func, "__qualname__") else str(func)
            )
            start = time.perf_counter() if logger.isEnabledFor(logging.DEBUG) else None
            out = func(*args, **kwargs)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Operation '%s' executed in %.6f",
                    func_name,
                    time.perf_counter() - start,
                )

            op_invoke_info = OpInvokeInfo(func, args, kwargs, out)
            self.op_invoke_infos.append(op_invoke_info)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Recorded '%s': %s", func_name, op_invoke_info)

            return out
        else:
            return func(*args, **kwargs)

    def repeat_op_invoke_infos(self):
        region_id_to_op_invoke_infos = {}
        current_id = None
        for op_invoke_info in self.op_invoke_infos:
            if (
                op_invoke_info.func
                == torch.ops.tensor_cast._internal_mark_region_begin.default
            ):
                assert current_id is None, (
                    f"Already in region {current_id}, we do not support nested regions"
                )
                current_id = op_invoke_info.args[1]
                assert current_id not in region_id_to_op_invoke_infos, (
                    f"Duplicated region id {current_id} found"
                )
                region_id_to_op_invoke_infos[current_id] = Region(op_invoke_info)
            elif (
                op_invoke_info.func
                == torch.ops.tensor_cast._internal_mark_region_end.default
            ):
                current_id = op_invoke_info.args[1]
                assert current_id in region_id_to_op_invoke_infos, (
                    f"Region end with id {current_id} not paired with a region begin"
                )
                region_id_to_op_invoke_infos[current_id].finalize(op_invoke_info)
                self.op_info_group.append(region_id_to_op_invoke_infos[current_id])
                current_id = None
            elif (
                op_invoke_info.func
                == torch.ops.tensor_cast._internal_copy_region.default
            ):
                assert current_id is None, (
                    f"Already in region {current_id}, we do not support nested regions"
                )
                copy_id = op_invoke_info.args[1]
                assert copy_id in region_id_to_op_invoke_infos, (
                    f"Regioin {copy_id} not marked before copy"
                )
                self.op_info_group.append(
                    region_id_to_op_invoke_infos[copy_id].shallow_copy(
                        op_invoke_info.args[0], op_invoke_info.out
                    )
                )
            else:
                if current_id is not None:
                    region_id_to_op_invoke_infos[current_id].op_invoke_infos.append(
                        op_invoke_info
                    )
                else:
                    self.op_info_group.append(op_invoke_info)

    def _replay_single_op(self, op_invoke_info):
        perf_results = {}
        for perf_model in self.perf_models:
            result = perf_model.process_op(op_invoke_info)
            perf_results[perf_model.name] = result
        self.event_list.append(
            RuntimeEvent(op_invoke_info=op_invoke_info, perf_results=perf_results)
        )

    def replay_op_invoke_infos(self):
        for op_info in self.op_info_group:
            if self.memory_tracker:
                self.memory_tracker.record_op_invocation(op_info)
            if isinstance(op_info, Region):
                for op_invoke_info in op_info.op_invoke_infos:
                    self._replay_single_op(op_invoke_info)
            else:
                self._replay_single_op(op_info)

    def __enter__(self):
        super().__enter__()
        self.exit_stack.enter_context(patch_torch())
        _current_runtime.value = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.repeat_op_invoke_infos()
        self.replay_op_invoke_infos()
        if self.memory_tracker:
            self.memory_tracker.analyze()
        _current_runtime.value = None

    def table_averages(self, group_by_input_shapes=False) -> str:
        """
        Dump pretty-print table, grouped by ops by default.

        TODO: consider to separate the data (event_list) and view (table)

        Args:
            group_by_input_shapes: group the events by input shapes when turned on.
        """
        if not self.event_list:
            return "No events recorded."

        def _format_time(seconds: float) -> str:
            """Formats time in seconds to a human-readable string (ms, us, ns)."""
            if seconds >= 1.0:
                return f"{seconds:.3f}s"
            if seconds >= 1e-3:
                return f"{seconds * 1e3:.3f}ms"
            if seconds >= 1e-6:
                return f"{seconds * 1e6:.3f}us"
            return f"{seconds * 1e9:.3f}ns"

        def _get_input_shapes_str(op_info: "OpInvokeInfo") -> str:
            """Extracts tensor shapes from operator arguments for display."""
            shapes = []
            # A simple search for tensors in args; can be expanded for kwargs or nested structures
            for arg in op_info.args:
                if isinstance(arg, torch.Tensor):
                    shapes.append(str(list(arg.shape)))
            return ", ".join(shapes)

        # --- Data Aggregation ---
        # The key is either the op name or (op name, input shapes)
        # The value stores the aggregated metrics for that key.
        aggregated_data = collections.defaultdict(
            lambda: {
                "count": 0,
                "total_runtimes": collections.defaultdict(float),
            }
        )

        for event in self.event_list:
            op_name = str(event.op_invoke_info.func)
            if group_by_input_shapes:
                shapes_str = _get_input_shapes_str(event.op_invoke_info)
                key = (op_name, shapes_str)
            else:
                key = op_name

            entry = aggregated_data[key]
            entry["count"] += 1
            for model_name, result in event.perf_results.items():
                entry["total_runtimes"][model_name] += result.execution_time_s

        if not aggregated_data:
            return "No performance results to display."

        # --- Prepare for Formatting ---
        model_names = [model.name for model in self.perf_models]

        # Sort entries by the total time of the first performance model, descending
        # This brings the most expensive operations to the top.
        first_model = model_names[0] if model_names else None

        def sort_key(item):
            if first_model:
                return item[1]["total_runtimes"].get(first_model, 0)
            return 0

        sorted_items = sorted(aggregated_data.items(), key=sort_key, reverse=True)

        # --- Define Headers and Calculate Column Widths ---
        headers = ["Name"]
        if group_by_input_shapes:
            for name in model_names:
                headers.append(f"{name} total")
            headers.append("Input Shapes")
        for name in model_names:
            headers.extend([f"{name} total", f"{name} avg"])
        headers.append("# of Calls")

        # Initialize widths with header lengths
        col_widths = {h: len(h) for h in headers}

        # Update widths based on data length
        for key, data in sorted_items:
            # Update Name column width
            name_str = key if isinstance(key, str) else key[0]
            col_widths["Name"] = max(col_widths["Name"], len(name_str))

            if group_by_input_shapes:
                shapes_str = key[1]
                col_widths["Input Shapes"] = max(
                    col_widths["Input Shapes"], len(shapes_str)
                )
                for model_name in model_names:
                    total_time_str = _format_time(data["total_runtimes"][model_name])
                    col_widths[f"{model_name} total"] = max(
                        col_widths[f"{model_name} total"], len(total_time_str)
                    )
            col_widths["# of Calls"] = max(
                col_widths["# of Calls"], len(str(data["count"]))
            )
            for model_name in model_names:
                total_time = data["total_runtimes"][model_name]
                avg_time = total_time / data["count"]
                col_widths[f"{model_name} total"] = max(
                    col_widths[f"{model_name} total"], len(_format_time(total_time))
                )
                col_widths[f"{model_name} avg"] = max(
                    col_widths[f"{model_name} avg"], len(_format_time(avg_time))
                )

        # --- Build Table String ---
        output_lines = []

        # Create header and separator lines
        header_line = "  ".join(h.center(col_widths[h]) for h in headers)
        separator_line = "  ".join("-" * col_widths[h] for h in headers)

        output_lines.append(separator_line)
        output_lines.append(header_line)
        output_lines.append(separator_line)

        # Create data rows
        for key, data in sorted_items:
            row = []
            name_str = key if isinstance(key, str) else key[0]
            row.append(name_str.ljust(col_widths["Name"]))

            if group_by_input_shapes:
                for model_name in model_names:
                    total_time_str = _format_time(data["total_runtimes"][model_name])
                    row.append(total_time_str.rjust(col_widths[f"{model_name} total"]))
                shapes_str = key[1]
                row.append(shapes_str.ljust(col_widths["Input Shapes"]))
            for model_name in model_names:
                total_time = data["total_runtimes"][model_name]
                avg_time = total_time / data["count"]
                row.append(
                    _format_time(total_time).rjust(col_widths[f"{model_name} total"])
                )
                row.append(
                    _format_time(avg_time).rjust(col_widths[f"{model_name} avg"])
                )
            row.append(str(data["count"]).rjust(col_widths["# of Calls"]))

            output_lines.append("  ".join(row))

        output_lines.append(separator_line)

        # --- Add Summary Footer ---
        summary_totals = collections.defaultdict(float)
        for data in aggregated_data.values():
            for model_name, total_time in data["total_runtimes"].items():
                summary_totals[model_name] += total_time

        for model_name in model_names:
            total_str = _format_time(summary_totals[model_name])
            output_lines.append(f"Total time for {model_name}: {total_str}")

        return "\n".join(output_lines)

    def get_trace_events(self):
        """
        Transform self.event_list to trace_events. Results from different performance models are
        arranged in different processes. Multiple streams are organized as threads in each process.
        """
        trace_events = []

        # Map performance model names to Process IDs (pid)
        perf_model_pids = {model.name: i for i, model in enumerate(self.perf_models)}

        # Keep track of the current time for each process/thread combination.
        # The key is the pid, and the value is the cumulative time in microseconds.
        # For now, we assume a single thread (tid=0) per process.
        current_time_us = dict.fromkeys(perf_model_pids.values(), 0)

        # 1. Add Metadata Events to name the processes for readability in the trace viewer
        for model_name, pid in perf_model_pids.items():
            trace_events.append(
                {
                    "name": "process_name",
                    "ph": "M",  # Metadata event type
                    "pid": pid,
                    "args": {"name": f"{model_name} (PID: {pid})"},
                }
            )
            # Also name the default thread for this process
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": 0,  # Assuming a single stream for now
                    "args": {"name": "Stream 0"},
                }
            )

        # 2. Iterate through events and create trace entries
        for event in self.event_list:
            op_name = str(event.op_invoke_info.func)

            # Create a trace event for each performance model's result
            for model_name, result in event.perf_results.items():
                pid = perf_model_pids[model_name]

                # result.runtime is in seconds, Chrome Trace wants microseconds
                duration_us = max(0, int(round(result.execution_time_s * 1e6)))

                # The event starts at the current cumulative time for this process
                start_time_us = current_time_us[pid]

                trace_event = {
                    "name": op_name,
                    "cat": model_name,  # Category can be the model name
                    "ph": "X",  # 'X' denotes a "complete" event (start and end time)
                    "ts": start_time_us,
                    "dur": duration_us,
                    "pid": pid,
                    "tid": 0,  # Hardcoded to thread 0 for now
                    "args": {  # Add any extra useful info here
                        "Inputs": str(event.op_invoke_info.args)
                        + " kwargs: "
                        + str(event.op_invoke_info.kwargs),
                        "Output": str(event.op_invoke_info.out),
                        **{
                            name: str(value)
                            for name, value in result.statistics.items()
                        },
                    },
                }
                trace_events.append(trace_event)

                # Update the cumulative time for this process's timeline
                current_time_us[pid] += duration_us

        return trace_events

    def export_chrome_trace(self, trace_file):
        """
        Dump trace_events as the chrome trace file.
        """
        trace_events = self.get_trace_events()
        # Write the final JSON object to the specified file

        if isinstance(trace_file, str):
            f = open(trace_file, "w")  # noqa: SIM115
            file_context = f
        else:
            f = trace_file
            file_context = contextlib.nullcontext()
        with file_context:
            # The top-level object should contain the 'traceEvents' key
            json.dump({"traceEvents": trace_events}, f)

    def get_breakdowns(self) -> Dict[str, Dict[str, float]]:
        """
        A breakdown of op categories according to the classification of each performance model in the runtime.
        The classification is decided by the performance models.

        Return:
            Dict: name of breakdown -> [category name, value for this category]
            The semantics of the values are defined by the performance models. See [NOTE: Breakdown from Op Classifier]
            for details.
            The runtime combines all the breakdowns from the classifiers of perf models.
        """
        breakdowns = {}
        for perf_model in self.perf_models:
            if classifiers := perf_model.get_classifiers():
                event_list_for_this = [
                    (event.op_invoke_info, event.perf_results[perf_model.name])
                    for event in self.event_list
                ]
                for classifier in classifiers:
                    breakdown = classifier.classify(event_list_for_this)
                    breakdowns[f"{perf_model.name}_{classifier.name}"] = breakdown
        return breakdowns

    def total_execution_time_s(self) -> Dict[str, float]:
        total: Dict[str, float] = {}
        for perf_model in self.perf_models:
            total[perf_model.name] = sum(
                event.perf_results[perf_model.name].execution_time_s
                for event in self.event_list
            )
        return total
