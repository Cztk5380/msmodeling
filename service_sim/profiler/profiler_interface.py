# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import sys
import os
import time

profiling_supported = True
try:
    from service_sim.profiler.profiler_stime import Level, SimProfiler, parse_main_func
except ImportError as e:
    profiling_supported = False

if profiling_supported:
    from service_sim.profiler.profiler_utils import (
        get_state,
        get_batch_type,
        get_iter_size_info,
        record_kv_cache_free_blocks,
        queue_profiler,
    )


def is_profiling_ready():
    return profiling_supported


def init_profiling():
    if not profiling_supported:
        raise Exception("profiling is not supported")
    prof = SimProfiler(Level.INFO).add_meta_info("service_type", "liuren_simulation")


def parse_profiling_results(profiling_path_with_timestamp):
    if not profiling_supported:
        raise Exception("profiling is not supported")
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

    sys.argv = ["python -m ms_service_profiler.parse", "--input-path", profiling_path_with_timestamp,\
        "--output-path", profiling_path_with_timestamp + "_parsed_result"]
    parse_main_func()