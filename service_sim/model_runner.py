# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import multiprocessing as mp
import queue
import threading
from multiprocessing import Event, Manager
from typing import List, Optional

import stime
from service_sim.config import Config
from service_sim.request import Request, RequestState

from tensor_cast.interface.model_runner import (
    ModelRunner as TensorCastModelRunner,
    ModelRunnerMetrics,
    RequestInfo,
)

logger = stime.get_logger(__name__)


class ModelRunner:
    def __init__(self, parallel_config, device_type, dp_rank: int):
        """
        Each model runner works on a data-parallel partition ('dp_rank') given a list of devices
        to compute the model given the model configuration 'model_config'. The computation is
        further sharded among the 'devices' via other parallel algorithms like tensor parallel etc.
        It instantiates a list of 'Workers' with each working on a device.
        """
        self.common_config = Config.get_instance().common_config
        self.parallel_config = parallel_config
        self.tensor_cast_model_runner = self.init_tensor_cast_model_runner(
            self.common_config, self.parallel_config, device_type
        )
        self.enable_multi_process = self.common_config.model_config.enable_multi_process
        self.num_processes = self.common_config.model_config.num_processes
        self.predict_steps = self.common_config.model_config.predict_steps
        if self.enable_multi_process:
            if not (isinstance(self.predict_steps, int) and self.predict_steps > 1):
                raise ValueError(
                    "check common_config.model_config.predict_steps, need to be int and > 1"
                )
            if isinstance(self.num_processes, int) and self.num_processes > 1:
                self.async_task_manager = AsyncTaskManager(
                    device_type, parallel_config, self.num_processes
                )
            else:
                raise ValueError(
                    "check common_config.model_config.num_processes, need be int and greater than 1"
                )

    @staticmethod
    def init_tensor_cast_model_runner(common_config, parallel_config, device_type):
        tensor_cast_model_runner = TensorCastModelRunner(
            device=device_type,
            model_id=common_config.model_config.name,
            do_compile=common_config.model_config.do_compile,
            allow_graph_break=common_config.model_config.allow_graph_break,
            dump_input_shapes=common_config.model_config.dump_input_shapes,
            chrome_trace=common_config.model_config.chrome_trace,
            quantize_linear_action=common_config.model_config.quantize_linear_action,
            quantize_lmhead=common_config.model_config.quantize_lmhead,
            mxfp4_group_size=common_config.model_config.mxfp4_group_size,
            quantize_attention_action=common_config.model_config.quantize_attention_action,
            num_mtp_tokens=common_config.model_config.num_mtp_tokens,
            num_hidden_layers_override=0,
            world_size=parallel_config.world_size,
            tp_size=parallel_config.tp_size,
            dp_size=parallel_config.dp_size,
            mlp_tp_size=parallel_config.mlp_tp_size,
            mlp_dp_size=parallel_config.mlp_dp_size,
            lmhead_tp_size=parallel_config.lmhead_tp_size,
            lmhead_dp_size=parallel_config.lmhead_dp_size,
            ep=parallel_config.ep,
            reserved_memory_gb=0.0,
            block_size=common_config.serving_config.block_size,
        )
        return tensor_cast_model_runner

    @staticmethod
    def request2info(batch: List[Request]) -> List[RequestInfo]:
        transferred_batch = []
        for req in batch:
            query_len = req.query_len
            num_input_tokens = req.num_input_tokens
            num_output_tokens = req.num_output_tokens
            seq_len = req.seq_len
            if not query_len <= seq_len:
                raise ValueError(
                    f"req_id: {req.id}, query_len {query_len} > seq_len {seq_len}"
                )
            if req.state not in [RequestState.PREFILLING, RequestState.DECODING]:
                raise ValueError(
                    f"req_id: {req.id}, state {req.state} is not PREFILLING or DECODING"
                )
            is_decode = req.state == RequestState.DECODING
            request_info = RequestInfo(
                query_len=query_len,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                seq_len=seq_len,
                is_decode=is_decode,
            )
            transferred_batch.append(request_info)

        return transferred_batch

    @staticmethod
    def predict_next_batch(current_batch: List[RequestInfo]) -> List[RequestInfo]:
        future_batch = []
        for current_req_info in current_batch:
            if current_req_info.seq_len < current_req_info.num_input_tokens:
                future_query_len = (
                    current_req_info.num_input_tokens - current_req_info.query_len
                )
                future_seq_len = current_req_info.seq_len
                future_is_decode = False
            elif (
                current_req_info.seq_len
                < current_req_info.num_input_tokens
                + current_req_info.num_output_tokens
                - 1
            ):
                future_query_len = 1  # TOBEDONE consider mtp here
                future_seq_len = current_req_info.seq_len + future_query_len
                future_is_decode = True
            elif (
                current_req_info.seq_len
                == current_req_info.num_input_tokens
                + current_req_info.num_output_tokens
                - 1
            ):
                continue
            else:
                raise ValueError(
                    "predict_next_batch: seq_len should not greater than num_input_tokens + num_output_tokens"
                )

            future_req_info = RequestInfo(
                query_len=future_query_len,
                num_input_tokens=current_req_info.num_input_tokens,
                num_output_tokens=current_req_info.num_output_tokens,
                seq_len=future_seq_len,
                is_decode=future_is_decode,
            )
            future_batch.append(future_req_info)
        return future_batch

    def process_batch(self, batch: List[Request]):
        batch = self.request2info(batch)
        if self.enable_multi_process:
            future_batch_list = []
            current_batch = batch
            for _ in range(self.predict_steps):
                future_batch = self.predict_next_batch(current_batch)
                if not future_batch:
                    break
                future_batch_list.append(future_batch)
                current_batch = future_batch
            result = self.async_task_manager.find_result(batch)
            if result is not None:
                duration = result.execution_time_s
            else:
                duration = self._get_estimated_time(batch)
            for future_batch in future_batch_list:
                self.async_task_manager.add_task(future_batch)
        else:
            duration = self._get_estimated_time(batch)

        with stime.Duration(duration):
            logger.debug(
                f"{self.common_config.model_config.name} process batch, batch length: {len(batch)}, "
                f"consume {duration} seconds"
            )

    def warmup(self) -> int:
        """
        use max length to try warmup get num_blocks
        """
        serving_config = Config.get_instance().common_config.serving_config
        batch = [
            RequestInfo(
                seq_len=serving_config.max_tokens_budget,
                query_len=serving_config.max_tokens_budget,
                is_decode=False,
                num_input_tokens=serving_config.max_tokens_budget,
                num_output_tokens=2 * serving_config.max_tokens_budget,
            )
        ]
        inference_metrics = self.tensor_cast_model_runner.run_inference(batch)
        block_size = self.common_config.serving_config.block_size
        all_mem_for_kv_cache = (
            inference_metrics.device_memory_available_gb
            + inference_metrics.kv_cache_size_gb
        )
        num_blocks = int(
            all_mem_for_kv_cache / inference_metrics.kv_cache_per_token_gb // block_size
        )
        logger.debug(f"warmup result: {num_blocks} blocks")
        return num_blocks, block_size

    def shutdown(self):
        if hasattr(self, "async_task_manager") and self.enable_multi_process:
            self.async_task_manager.shutdown()

    def _get_estimated_time(self, batch):
        estimated_time = self.tensor_cast_model_runner.run_inference(
            batch
        ).execution_time_s
        return estimated_time


class AsyncTask:
    def __init__(self, batch: List[RequestInfo]):
        self.batch = batch
        self.hash_value = self.get_hash()

    def get_hash(self) -> str:
        return hash(tuple(tuple(vars(req_info).items()) for req_info in self.batch))


class ModelRunnerMetricCacheManager:
    def __init__(self, multiprocessing_manager):
        self.cache = multiprocessing_manager.dict()

    def init_cache_slot(self, cache_id: str) -> None:
        """main process call this func: init cache slot for new cache"""
        if cache_id in self.cache:
            raise ValueError(
                f"init_task_slot failed, cache with cache_id {cache_id} already exists"
            )

        self.cache[cache_id] = None

    def get_cache(self, cache_id: str) -> Optional[ModelRunnerMetrics]:
        """main process call this func: get cache by cache_id"""
        if cache_id not in self.cache:
            raise KeyError(
                f"get_cache failed, cache with cache_id {cache_id} not found"
            )

        return self.cache[cache_id]

    def record_cache(self, cache_id: str, result: ModelRunnerMetrics) -> None:
        """child process call this func: record cache"""
        if cache_id not in self.cache:
            raise KeyError(
                f"record_cache failed, cache with cache_id {cache_id} not found"
            )

        self.cache[cache_id] = result


class CompletionEventManager:
    def __init__(self, multiprocessing_manager):
        self.event_dict = {}
        self.completion_queue = multiprocessing_manager.Queue()

        self._thread_running = True

        self._event_thread = threading.Thread(
            target=self._process_completion_queue, daemon=True
        )
        self._event_thread.start()



    def init_event_slot(self, event_id):
        """main process call this func: init event slot for new task"""
        if event_id in self.event_dict:
            raise ValueError(
                f"init_event_slot failed, event with event_id {event_id} already exists"
            )

        event = Event()
        event.clear()
        self.event_dict[event_id] = event

    def wait_completion_event(self, event_id):
        """main process call this func: wait event by event_id"""
        self.event_dict[event_id].wait()

    def set_completion_event(self, event_id):
        """child process call this func: put completion message into queue and wait thread in main process to deal"""
        self.completion_queue.put(event_id)

    def shutdown(self) -> None:
        """
        Safely shut down the event manager, ensuring background thread exits and resources are cleaned up.
        """
        # Stop the thread's running loop
        self._thread_running = False
        logger.info("CompletionEventManager: Notifying background thread to stop")

        # Wait for the background thread to exit
        if self._event_thread.is_alive():
            logger.info("CompletionEventManager: Waiting for background thread to exit")
            # Put a dummy value to wake up the thread if it's blocked on get()
            self.completion_queue.put(None)
            self._event_thread.join(timeout=5)  # Wait at most 5 seconds
            if self._event_thread.is_alive():
                logger.warning(
                    "CompletionEventManager: Warning - Background thread did not exit in time"
                )

        # Clean up remaining items in the completion queue
        logger.info(
            "CompletionEventManager: Clearing remaining items in completion queue"
        )
        try:
            while not self.completion_queue.empty():
                # Remove all remaining elements (including the dummy None if present)
                self.completion_queue.get_nowait()
                self.completion_queue.task_done()
        except Exception as e:
            logger.error(
                f"CompletionEventManager: Error while clearing queue - {str(e)}"
            )

        # Clear event dictionary to release resources
        self.event_dict.clear()
        logger.info("CompletionEventManager: All resources cleaned up")

    def _process_completion_queue(self) -> None:
        """main process start this thread: get completion message and do event.set()"""
        while self._thread_running:
            try:
                event_id = self.completion_queue.get(timeout=1)
            except queue.Empty:
                continue

            if event_id is None:
                continue

            if event_id in self.event_dict:
                event = self.event_dict[event_id]
                event.set()
            else:
                raise ValueError
            self.completion_queue.task_done()


class AsyncTaskManager:
    def __init__(self, device_type, parallel_config, num_workers: int = 2):
        self.manager = Manager()

        self.task_queue = self.manager.Queue()
        self.model_runner_metrics_cache_manager = ModelRunnerMetricCacheManager(
            self.manager
        )
        self.event_manager = CompletionEventManager(self.manager)

        self.stop_event = mp.Event()

        self.task_record = set()

        # init multi processes
        self.num_workers = num_workers
        self.workers = []
        self._init_multi_process(device_type, parallel_config)

    def add_task(self, batch: List[RequestInfo]) -> None:
        task = AsyncTask(batch)
        task_hash = task.hash_value
        if task_hash not in self.task_record:
            self.model_runner_metrics_cache_manager.init_cache_slot(task_hash)
            self.event_manager.init_event_slot(task_hash)
            self.task_record.add(task_hash)
            self.task_queue.put(task)

    def find_result(self, batch: List[RequestInfo]):
        task = AsyncTask(batch)
        task_hash = task.hash_value
        if task_hash in self.task_record:
            self.event_manager.wait_completion_event(task_hash)
            result = self.model_runner_metrics_cache_manager.get_cache(task_hash)
        else:
            result = None

        return result

    def shutdown(self) -> None:
        self.stop_event.set()
        logger.debug("Shutdown: stop event set, workers will exit loop")

        for idx, p in enumerate(self.workers):
            p.join(timeout=15)
            if p.is_alive():
                logger.warning(f"Worker {idx} not exit in time, terminating")
                p.terminate()
                p.join(timeout=5)
            logger.debug(f"Worker {idx} exited")
        self.workers.clear()

        self.event_manager.shutdown()

        self.manager.shutdown()
        logger.debug("Shutdown: Manager closed")

    def _init_multi_process(self, device_type, parallel_config) -> None:
        def worker(barrier):
            try:
                common_config = Config.get_instance().common_config
                tensor_cast_model_runner = ModelRunner.init_tensor_cast_model_runner(
                    common_config, parallel_config, device_type
                )
                barrier.wait()  # ensure all processes have built the model
            except Exception as e:
                logger.error(f"Worker initialization failed: {str(e)}")
                return

            while not self.stop_event.is_set():
                try:
                    task = self.task_queue.get(timeout=1)
                    task_hash = task.hash_value
                except queue.Empty:
                    continue
                except Exception as e:
                    if self.stop_event.is_set():
                        logger.debug("Worker exiting, stop event set")
                        break
                    raise RuntimeError("AsyncTaskManager get task failed") from e

                try:
                    result = tensor_cast_model_runner.run_inference(task.batch)
                except Exception as e:
                    raise RuntimeError("AsyncTaskManager execute task failed") from e

                self.model_runner_metrics_cache_manager.record_cache(task_hash, result)
                self.event_manager.set_completion_event(task_hash)

        barrier = mp.Barrier(self.num_workers + 1)
        for _ in range(self.num_workers):
            p = mp.Process(target=worker, args=(barrier,), daemon=True)
            p.start()
            self.workers.append(p)
        barrier.wait()