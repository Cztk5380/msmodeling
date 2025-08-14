# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import threading
from typing import Dict, List

import stime
from device import Device
from model import ModelConfig
from model_runner import ModelRunner
from request import Request, RequestState


logger = stime.get_logger(__name__)


class BatchScheduler:
    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.request_queue = stime.Queue(allow_anti_causality_put=True)
        self.batch_queue = stime.Queue()
        self._shutdown = threading.Event()
        self.batching_timeout = 0
        self.batching_thread = stime.Thread(target=self._batching_loop, daemon=True)
        self.runner_thread = stime.Thread(target=self._runner_loop, daemon=True)
        self.requests: Dict[int, Request] = {}
        self.requests_condition = stime.Condition()

    def add(self, request: Request):
        logger.debug(f"BatchScheduler adding {request}")
        self.request_queue.put(request)
        with self.requests_condition:
            self.requests[request.id] = request

    def run(self):
        self.batching_thread.start()
        self.runner_thread.start()

    def shutdown(self):
        """Gracefully stop the processing thread."""
        self._shutdown.set()
        self.request_queue.shutdown()
        self.batch_queue.shutdown()
        self.batching_thread.join()  # wait for the thread to finish
        self.runner_thread.join()

    def get_work_load(self):
        res = 0
        with self.requests_condition:
            for request in self.requests.values():
                if request.state == RequestState.PREFILLING:
                    res += request.num_input_tokens
                elif request.state == RequestState.DECODING:
                    res += 1
        return res

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
        """Threading target: collect requests into batches and process them."""
        try:
            while not self._shutdown.is_set():
                batch = self._collect_batch()
                if batch:
                    # put the batch into the batch queue for the runner thread to pick up
                    logger.debug(
                        f"Collected batch size: {len(batch)} {[request.id for request in batch]}"
                    )
                    self.batch_queue.put(batch)
        except:
            logger.exception("Unexpected exception in the batching loop")
            raise

    def _postprocess_batch(self, batch: List[Request]):
        """
        Mark requests done and release resources
        Put incomplete requests back into the queue
        """
        continuous_batching_requests = []
        for request in batch:
            if (
                request.state != RequestState.PREFILLING
                and request.state != RequestState.DECODING
            ):
                raise ValueError(
                    "In _postprocess_batch, request.state should be PREFILLING or DECODING,"
                    " but get %s",
                    request.state,
                )
            request.num_decoded_tokens += 1
            if request.num_decoded_tokens >= request.num_output_tokens:
                with self.requests_condition:
                    self.requests.pop(request.id)
                    self.requests_condition.notify_all()
                request.state = RequestState.DECODE_DONE

            if request.state == RequestState.PREFILLING:
                with self.requests_condition:
                    self.requests.pop(request.id)
                    self.requests_condition.notify_all()
                request.state = RequestState.PREFILL_DONE
            elif request.state == RequestState.DECODING:
                continuous_batching_requests.append(request)

        self.request_queue.put_items(continuous_batching_requests)

    def _runner_loop(self):
        try:
            while not self._shutdown.is_set():
                batch = self.batch_queue.get()
                if batch is None:
                    raise ValueError("In _runner_loop, batch is None")
                self.model_runner.process_batch(batch)  # Process the batch
                self._postprocess_batch(batch)
        except:
            logger.exception("Unexpected exception in the runner loop")
            raise


class Engine:
    """
    Process request, PREFILLING --> PREFILL_DONE or DECODING --> DECODE_DONE
    """

    def __init__(self, devices: List[Device], dp_rank: int, model_config: ModelConfig):
        self.model_runner = ModelRunner(devices, dp_rank, model_config)
        self.batch_scheduler = BatchScheduler(self.model_runner)
        self.batch_scheduler.run()

    def handle(self, request: Request):
        logger.debug(f"Engine handling {request}")
        if request.state not in [RequestState.PREFILLING, RequestState.DECODING]:
            raise ValueError(
                "Engine.handle failed, request.state should be PREFILLING or DECODING, "
                "but get request.state: %s",
                request.state,
            )
        self.batch_scheduler.add(request)

    def get_work_load(self) -> int:
        """
        work_load is an abstract score using to measure the inference work of engine
        """
        return self.batch_scheduler.get_work_load()


class EngineLoadBalancer:
    def __init__(self, engines: List[Engine]):
        self.engines = engines

    def select(self, request: Request) -> Engine:
        # greedily choose the instance having the least total number of input tokens to handle
        # TOBEDONE: we should expose metrics from the engine instead of exposing all the request instances
        # TOBEDONE: support heterogeneous instances
        work_loads = [engine.get_work_load() for engine in self.engines]
        min_value = min(work_loads)
        min_index = work_loads.index(min_value)
        return self.engines[min_index]
