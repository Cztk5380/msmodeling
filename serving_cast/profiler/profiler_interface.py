# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import sys
import time

profiling_supported = True
try:
    from serving_cast.profiler.profiler_stime import Level, parse_main_func, SimProfiler
except ImportError:
    profiling_supported = False


def is_profiling_ready():
    return profiling_supported


def init_profiling():
    if not profiling_supported:
        raise ValueError("profiling is not supported")
    _ = SimProfiler(Level.INFO).add_meta_info("service_type", "liuren_simulation")


def parse_profiling_results(profiling_path_with_timestamp):
    if not profiling_supported:
        raise ValueError("profiling is not supported")
    # wait till profiling collect done
    last = None
    same = 0
    stable_cnt = 10
    while True:
        total = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(profiling_path_with_timestamp)
            for f in files
        )
        if total == last:
            same += 1
            if same >= stable_cnt:
                break
        else:
            same = 0
            last = total
        time.sleep(0.1)

    sys.argv = [
        "python -m ms_service_profiler.parse",
        "--input-path",
        profiling_path_with_timestamp,
        "--output-path",
        profiling_path_with_timestamp + "_parsed_result",
    ]
    parse_main_func()
