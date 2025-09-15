# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import stime
from stime import get_logger, Task

logger = get_logger(__name__)


def main_processing(serving, load_gen):
    while load_gen.has_request():
        request, interval = load_gen.next_request()
        serving.serve(request)
        stime.elapse(interval)
    while not load_gen.is_finished():
        stime.elapse(10)
            
    logger.debug(f"time {stime.now():.1f}: all of the requests are finished, stop simulation")
    stime.stop_simulation()
    return
            