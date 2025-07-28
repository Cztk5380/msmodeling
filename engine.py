# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import math
import time
from typing import Dict, List
import threading

import stime
from device import Device
from model import ModelConfig
from model_runner import ModelRunner
from request import Request, RequestState

logger = stime.getLogger(__name__)


class BatchScheduler:
    def __init__(self, model_runner: ModelRunner):
        self. model_runner = model_runner
        self.request_queue = stime.Queue(allow_anti_causality_put=True)
        self.batch_queue = stime.Queue()
        self._shutdown = threading.Event()
        self.batching_timeout = 0

    def add(self, request: Request):
        logger.debug(f"BatchScheduler adding {request}")
        self.request_queue.put(request)

    def run(self):
        self.batching_thread = stime.Thread(target=self._batching_loop, daemon=True)
        self.batching_thread.start()
        self.runner_thread = stime.Thread(target=self._runner_loop, daemon=True)
        self.runner_thread.start()

    def _collect_batch(self):
        """Collect request batch and allocate resources for batched requests"""
        # Naively batch all the requests in the queue and assume resources are enough
        # TOBEDONE: use more complex batching policy

        logger.debug("Collecting batch")
        batch = self.request_queue.get_all_due()
        if not batch:
            now = stime.now()
            # move fast-forward num_timeout_waits times to get pass first_ts
            # this simulates the timeout wait and makes sure we get a batch
            # on the next collection call
            self.request_queue.wait_till_due(self.batching_timeout)
            logger.debug(
                f"No due requests, fast-forwarding from {now:.3f} to {stime.now():.3f}"
            )
        return batch
    
    def _batching_loop(self):
        """Threading target: collect requests inot batches and process them."""
        try:
            while not self._shutdown.is_set():
                batch = self._collect_batch()
                if batch:
                    # put the batch into the batch queue for the runner thread to pick up
                    logger.debug(f"Collected batch size: {len(batch)} {[request.id for request in batch]}")
                    self.batch_queue.put(batch)
        except:
            logger.error(f"Unexpected exception in the batching loop", exc_info=True)
            raise

    def _postprocess_batch(self, batch: List[Request]):
        """
        Mark requests done and release resources
        Put incomplete requests back into the queue
        """
        continuous_batching_requests = []
        for request in batch:
            if request.state != RequestState.PREFILLING and request.state != RequestState.DECODING:
                raise ValueError("request.state != RequestState.PREFILLING and request.state != RequestState.DECODING")
            request.num_decoded_tokens += 1
            if request.num_decoded_tokens >= request.num_output_tokens:
                request.state = RequestState.DECODE_DONE

            if request.state == RequestState.PREFILLING:
                request.state = RequestState.PREFILL_DONE
            elif request.state == RequestState.DECODING:
                continuous_batching_requests.append(request)
            
        self.request_queue.put_items(continuous_batching_requests)

    def _runner_loop(self):
        try:
            while not self._shutdown.is_set():
                batch = self.batch_queue.get()
                if batch is None:
                    raise ValueError("batch is None")
                # looger.debug(f"Processing {[request.id for request in batch]}"")
                self.model_runner.process_batch(batch) # Process the batch
                self._postprocess_batch(batch)
        except:
            logger.error(f"Unexpected exception in the runner loop", exc_info=True)
            raise

    def shutdown(self):
        """Gracefully stop the processing thread."""
        self._shutdown.set()
        self.request_queue.shutdown()
        self.batch_queue.shutdown()
        self.batching_thread.join() # wait for the thread to finish
        self.runner_thread.join()

    @property
    def requests(self) -> Dict[int, Request]:
        requests: Dict[int, Request] = {}
        for request in self.request_queue:
            requests[request.id] = request
        for batch in self.batch_queue:
            for request in batch:
                requests[request.id] = request
        return requests
    

class Engine:
    def __init__(self, devices: List[Device], dp_rank: int, model_config: ModelConfig):
        self.model_runner = ModelRunner(devices, dp_rank, model_config)
        self.batch_scheduler = BatchScheduler(self.model_runner)
        self.batch_scheduler.run()

    def handle(self, request: Request):
        logger.debug(f"Engine handling {request}")
        self.batch_scheduler.add(request)

    @property
    def requests(self) -> Dict[int, Request]:
        return self.batch_scheduler.requests


class PrefillEngineLoadBalacer:
    def __init__(self, engines: List[Engine]):
        self.engines = engines
    
    def select(self, request: Request) -> Engine:
        # greedily choose the instance having the least total number of input tokens to handle
        # TOBEDONE: we should expose metrics from the engine instead of exposing all the request instances
        # TOBEDONE: support heterogeneous instaces
        sums = [sum(request.num_input_tokens for request in engine.requests.values()) for engine in self.engines]
        min_value = min(sums)
        min_index = sums.index(min_value)
        return self.engines[min_index]
    

class DecodeEngineLoadBalancer:
    def __init__(self, engines: List[Engine]):
        self.engines = engines

    def select(self, request: Request) -> Engine:
        # greedily choose the instance having the least total number of requests to hand
        # TOBEDONE: we should expose metrics from the engine instead of exposing all the request instances
        # TOBEDONE: support heterogeneous instances
        sums = [len(engine.requests) for engine in self.engines]
        min_value = min(sums)
        min_index = sums.index(min_value)
        return self.engines[min_index]
    