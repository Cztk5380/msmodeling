# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import threading
from typing import Dict, List

import stime
from device import Device
from model import ModelConfig
from model_runner import ModelRunner
from request import Request, RequestState
from kv_cache_manager import KVCacheManager



logger = stime.get_logger(__name__)

MEM_LEFT_FOR_KV_CACHE = 40 * 1024 * 1024 # 40GB
MAX_TOKENS_BUDGET = 16384


class BatchScheduler:
    def __init__(self, model_runner: ModelRunner, kv_manager: KVCacheManager):
        self.model_runner = model_runner
        self.kv_manager = kv_manager
        self.waiting_queue = stime.Queue(allow_anti_causality_put=True)
        self.running_queue = stime.Queue()
        self.kv_sending_queue = stime.Queue()
        self._shutdown = threading.Event()
        self.batching_timeout = 0
        self.scheduling_thread = stime.Thread(target=self._scheduling_loop, daemon=True)
        self.requests: Dict[int, Request] = {}
        self.requests_condition = stime.Condition()
        self.max_tokens_budget = MAX_TOKENS_BUDGET

    def add(self, request: Request):
        logger.debug(f"BatchScheduler adding {request}")
        self.waiting_queue.put(request)
        with self.requests_condition:
            self.requests[request.id] = request
            self.requests_condition.notify_all()

    def run(self):
        self.scheduling_thread.start()

    def shutdown(self):
        """Gracefully stop the processing thread."""
        self._shutdown.set()
        self.waiting_queue.shutdown()
        self.running_queue.shutdown()
        self.scheduling_thread.join() # wait for the thread to finish

    def get_work_load(self):
        res = 0
        with self.requests_condition:
            for request in self.requests.values():
                if request.state == RequestState.PREFILLING:
                    res += request.num_input_tokens
                elif request.state == RequestState.DECODING:
                    res += 1
        return res

    def _schedule(self):
        req_index = 0
        token_budget = self.max_tokens_budget
        preempt_reqs = []

        # first schedule reqs in running_queue
        while req_index < len(self.running_queue) and token_budget > 0:
            request = self.running_queue[req_index]
            num_computed_tokens = min(token_budget, request.num_current_max_new_tokens)
            if num_computed_tokens <= 0:
                raise ValueError(
                    f"num_computed_tokens should be positive, got {num_computed_tokens}"
                )
            
            # try to allocate KV cache for the request
            while True:
                new_blocks = self.kv_manager.allocate_slots(request.id, num_computed_tokens)

                if new_blocks is None:
                    # The request cannot be scheduled. Preempt the lowest-priority request. now naively preempt the last
                    request_to_preempt = self.running_queue.peek_latest_due()
                    if request_to_preempt is not request: # not the current req
                        self.running_queue.remove(request_to_preempt)
                        self._process_preempted_request(request_to_preempt)
                        logger.debug("BatchScheduler._schedule: preempt request %s", request_to_preempt.id)
                        preempt_reqs.append(request_to_preempt)
                    else:
                        can_schedule = False
                        break
                else:
                    can_schedule = True
                    break
            if not can_schedule:
                break
            if new_blocks is None:
                raise ValueError("BatchScheduler._schedule failed: new_blocks should not be None")

            # kv cache allocation success, schedule the request
            token_budget -= num_computed_tokens
            request.num_current_max_new_tokens -= num_computed_tokens
            req_index += 1

        # second if no preempted requests (meaning kv cache has available slots), schedule the waiting_queue
        if not preempt_reqs:
            while self.waiting_queue and token_budget > 0:
                request = self.waiting_queue.peek_due()
                if not request:
                    break
                # check if need receive kv transfer, if need then try to receive, if failed skip
                if request.state == RequestState.KVS_TRANSFERRING and request.need_kv_transfer and \
                    not request.kv_transfer_done:
                    if not self._receive_remote_kvs(request):
                        continue
                num_computed_tokens = min(token_budget, request.num_current_max_new_tokens)
                if num_computed_tokens <= 0:
                    raise ValueError(
                        f"num_computed_tokens should be positive, got {num_computed_tokens}"
                    )
                new_blocks = self.kv_manager.allocate_slots(request.id, num_computed_tokens)
                # try to allocate kv cache, if failed, no need to check the rest requests in waiting queue
                if new_blocks is None:
                    logger.debug("BatchScheduler._schedule: Schedule request %s failed, due to lack of KV cache. " \
                        "KV manager status: %s", request, self.kv_manager.stats())
                    self.kv_manager.free(request.id)
                    break

                # kv cache allocation success, schedule the request
                token_budget -= num_computed_tokens
                request.num_current_max_new_tokens -= num_computed_tokens
                self.running_queue.put(request)
                self.waiting_queue.remove(request)

        if len(self.running_queue) == 0:
            logger.debug("BatchScheduler._schedule: no requests are scheduled, trigger wait_till_due")
            self.waiting_queue.wait_till_due()

    def _receive_remote_kvs(self, request) -> bool:
        transferred_num_tokens = request.num_input_tokens
        new_blocks = self.kv_manager.allocate_slots(request.id, transferred_num_tokens)
        if new_blocks is not None:
            request.kv_transfer_done = True
            request.state = RequestState.DECODING
            return True
        return False

    def _send_kvs_from_remote(self, request):
        self.running_queue.remove(request)
        self.kv_manager.free(request.id)
        request.state = RequestState.KVS_TRANSFERRING


    def _process_preempted_request(self, request: Request):
        """
        the request is currently in running_queue.
        need to move it to waiting_queue, change its state to WAITING,
        change its num_current_max_new_tokens, free its kvcache
        """
        self.running_queue.remove(request)
        self.waiting_queue.put(request)
        request._state = RequestState.PREFILLING
        request.num_current_max_new_tokens = request.num_input_tokens
        request.num_decoded_tokens = 0
        self.kv_manager.free(request.id)

    def _scheduling_loop(self):
        """
        Threading target: 
        First, schedule the requests into waiting_queue or running_queue.
        Second, execute the requests in the running_queue.
        """
        try:
            while not self._shutdown.is_set():
                self._schedule()
                if len(self.running_queue) != 0:
                    logger.debug(f"Scheduled batch size: {len(self.running_queue)}"
                        f"request ids: {[request.id for request in self.running_queue]}")
                    self.model_runner.process_batch(self.running_queue)
                    self._postprocess_batch()
        except:
            logger.error(f"Unexpected exception in the scheduling loop", exc_info=True)
            raise

    def _postprocess_batch(self):
        """
        Mark requests done and release resources
        Put incomplete requests back into the queue
        """
        idx = 0
        while idx < len(self.running_queue):
            request = self.running_queue[idx]
            if request.state != RequestState.PREFILLING and request.state != RequestState.DECODING:
                raise ValueError("In _postprocess_batch, request.state should be PREFILLING or DECODING," \
                    " but get %s" % request.state)
            if request.num_current_max_new_tokens == 0:
                # totally finish one step of prefilling or decoding
                request.num_decoded_tokens += 1
                if request.num_decoded_tokens >= request.num_output_tokens:
                    with self.requests_condition:
                        self.requests.pop(request.id)
                        self.requests_condition.notify_all()
                    self.kv_manager.free(request.id)
                    self.running_queue.remove(request)
                    request.state = RequestState.DECODE_DONE
                    continue

                request.num_current_max_new_tokens = 1
                if request.state == RequestState.PREFILLING:
                    request.state = RequestState.PREFILL_DONE
                    
                    if request.need_kv_transfer:
                        if request.kv_transfer_done:
                            raise ValueError("BatchScheduler._postprocess_batch failed: "
                                "request's kv cache should not been transferred")
                        self._send_kvs_from_remote(request)
                        with self.requests_condition:
                            self.requests.pop(request.id)
                            self.requests_condition.notify_all()
                        continue
                    request.state = RequestState.DECODING
            else:
                # case when prefill are chunked, nothing need to do right now
                logger.debug("requset %d are chunked, num of tokens need to compute left %d", request.id, \
                    request.num_current_max_new_tokens)
            idx += 1
    

class Engine:
    """
    Process request, PREFILLING --> PREFILL_DONE or DECODING --> DECODE_DONE
    """

    def __init__(self, devices: List[Device], dp_rank: int, model_config: ModelConfig):
        self.model_runner = ModelRunner(devices, dp_rank, model_config)
        self.kv_manager = self.create_kv_manager(model_config)
        self.batch_scheduler = BatchScheduler(self.model_runner, self.kv_manager)
        self.batch_scheduler.run()

    @staticmethod
    def create_kv_manager(model_config):
        head_dim = model_config.kwargs.get("head_dim")
        num_heads = model_config.kwargs.get("num_heads")
        precision_bytes = model_config.kwargs.get("precision_bytes")
        num_layers = model_config.kwargs.get("num_layers")
        if None in [head_dim, num_heads, precision_bytes, num_layers]:
            raise ValueError("head_dim, num_heads, precision_bytes and num_layers must be set")
        kv_cache_per_layer = MEM_LEFT_FOR_KV_CACHE // num_layers
        block_nums = kv_cache_per_layer // (2 * head_dim * num_heads * precision_bytes)
        kv_manager = KVCacheManager(block_nums)
        return kv_manager

    def handle(self, request: Request):
        logger.debug(f"Engine handling {request}")
        if request.state not in [RequestState.PREFILLING, RequestState.DECODING, RequestState.KVS_TRANSFERRING]:
            raise ValueError("Engine.handle failed, request.state should be PREFILLING, DECODING or "
                "KVS_TRANSFERRING but get request.state: %s" % request.state)
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
