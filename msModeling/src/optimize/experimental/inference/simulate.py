# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import atexit
import json
import os
import queue
import signal
import stat
import threading
import time
from pathlib import Path
from threading import Thread
from typing import Optional
from loguru import logger
import numpy as np
import torch
from loguru import logger

from msserviceprofiler.modelevalstate.config.config import get_settings
from msserviceprofiler.modelevalstate.inference.constant import IS_SLEEP_FLAG, BatchStage
from msserviceprofiler.modelevalstate.inference.data_format_v1 import BatchField, RequestField, ConfigPath
from msserviceprofiler.modelevalstate.inference.dataset import CustomLabelEncoder, preset_category_data, DataProcessor
from msserviceprofiler.modelevalstate.inference.file_reader import FileHanlder, StaticFile
from msserviceprofiler.modelevalstate.inference.state_eval_v1 import predict_v1_with_cache
from msserviceprofiler.msguard.security import open_s

predict_queue = queue.Queue()


class ServiceField:
    batch_field = None
    request_field = None
    next_tokens = None
    config_path = None
    fh = None
    data_processor = None
    req_id_and_max_decode_length = None


settings = get_settings()


ServiceField.config_path = ConfigPath(settings.latency_model.model_path,
                                      settings.latency_model.static_file_dir,
                                      settings.latency_model.req_and_decode_file,
                                      settings.latency_model.cache_data)

sub_thread: Optional[Thread] = None


class FileLogger:
    def __init__(self, file_path, mode="a"):
        self.file_path = file_path
        self.fout = None
        self.lock = threading.Lock()
        self.mode = mode

    def open_file(self):
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        if not isinstance(self.file_path, Path):
            self.file_path = Path(self.file_path)
        self.fout = os.fdopen(os.open(self.file_path, flags, modes), self.mode, buffering=1024)

    def write(self, message):
        with self.lock:
            if self.fout:
                self.fout.write(message)
                self.fout.write("\n")
                self.fout.flush()

    def close(self):
        with self.lock:
            if self.fout:
                self.fout.close()
                self.fout = None


def write_file(file_logger):
    file_logger.open_file()
    while True:
        flag = False
        items = []
        while not predict_queue.empty():
            items.append(predict_queue.get())
        for res in items:
            if res is None:
                flag = True
                break
            file_logger.write(str(res))
        if flag:
            break
        time.sleep(1)
    file_logger.close()


def signal_handler(file_logger):
    predict_queue.put(None)
    if sub_thread:
        sub_thread.join(timeout=3)
        if sub_thread.is_alive():
            raise TimeoutError("子线程未在指定时间完成")
    file_logger.close()


file_log = FileLogger(Path(settings.simulator_output).joinpath(f"simulate_{os.getpid()}.csv"))
atexit.register(signal_handler, file_log)


class Simulate:
    first = True
    predict_cache = {}
    request_seq_len = {}

    @staticmethod
    def init(plugin_object):
        if Simulate.first:
            if isinstance(plugin_object.input_manager.cache_config.eos_token_id, int):
                plugin_object.eos_token_id = plugin_object.input_manager.cache_config.eos_token_id
            else:
                plugin_object.eos_token_id = plugin_object.input_manager.cache_config.eos_token_id[0]
            if ServiceField.config_path.req_and_decode_file.exists():
                with open_s(ServiceField.config_path.req_and_decode_file, 'r') as f:
                    try:
                        req_and_decode = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in config file: {e}") from e
                    ServiceField.req_id_and_max_decode_length = {
                        int(k): int(v) 
                        for k, v in req_and_decode.items()
                        if str(k).strip().isdigit() and str(v).strip().isdigit()
                    }
            else:
                ServiceField.req_id_and_max_decode_length = {}
            if not Path(ServiceField.config_path.static_file_dir).exists():
                Path(ServiceField.config_path.static_file_dir).mkdir(parents=True, mode=0o750)
            static_file = StaticFile(base_path=ServiceField.config_path.static_file_dir)
            ServiceField.fh = FileHanlder(static_file)
            ServiceField.fh.load_static_data()
            custom_encoder = CustomLabelEncoder(preset_category_data)
            custom_encoder.fit()
            ServiceField.data_processor = DataProcessor(custom_encoder)
            Simulate.first = False
            global sub_thread
            sub_thread = threading.Thread(target=write_file, args=(file_log,))
            sub_thread.start()

    @staticmethod
    def generate_random_token(plugin_object, shape, max_value=32000):
        # max_value 是vacab size，就是词表的范围
        if np.prod(shape) > max_value + 1:
            raise ValueError("token数量超过词表的范围，无法进行无放回抽样")
        array = np.random.choice(np.arange(0, max_value + 1), size=np.prod(shape), replace=False)
        array = np.reshape(array, shape)
        array = np.where(array == plugin_object.eos_token_id, np.random.randint(0, max_value + 1), array)
        return array

    @staticmethod
    def generate_logits(batch_size, vocab_size: int = 129280, device="npu:0", dtype="float16"):
        dtype_map = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float: "float",
            torch.int8: "int8"
        }
        _cur_dtype = torch.float16
        for k, v in dtype_map.items():
            if v == dtype:
                _cur_dtype = k
                break
        tensor = torch.randn((batch_size, vocab_size), dtype=_cur_dtype, device=device)
        return tensor

    @staticmethod
    def generate_features(plugin_object, input_metadata, cached_ids):
        output_len_count = plugin_object.input_manager.cache.output_len_count[cached_ids]
        if input_metadata.is_prefill:
            batch_stage = BatchStage.PREFILL
            for i, req_id in enumerate(input_metadata.batch_request_ids):
                Simulate.request_seq_len[req_id] = input_metadata.batch_seq_len[i]
            all_input_length = input_metadata.batch_seq_len
        else:
            batch_stage = BatchStage.DECODE
            all_input_length = []
            for req_id in input_metadata.batch_request_ids:
                all_input_length.append(Simulate.request_seq_len[req_id])
        all_need_blocks = np.count_nonzero(input_metadata.block_tables > -1, axis=-1)
        request_field = []
        _total_req_input_len = []
        try:
            for _cache_id_index, _cache_id in enumerate(cached_ids):
                _req_input_len = all_input_length[_cache_id_index] - output_len_count[_cache_id_index]
                _total_req_input_len.append(_req_input_len)
                request_field.append(RequestField(_req_input_len,
                                                all_need_blocks[_cache_id_index],
                                                output_len_count[_cache_id_index]))
            batch_field = BatchField(batch_stage, input_metadata.batch_size,
                                    np.count_nonzero(input_metadata.block_tables > -1, axis=-1).sum(),
                                    sum(_total_req_input_len), max(_total_req_input_len))
        except Exception as e:
            logger.warning(f"process ServiceField in generate_features failed: {e}")
            ServiceField.batch_field = None
            ServiceField.request_field = None
            batch_field = None
            return batch_field, request_field

        request_field = tuple(sorted(request_field))
        ServiceField.batch_field = batch_field
        ServiceField.request_field = request_field
        return batch_field, request_field

    @staticmethod
    def update_token(plugin_object, input_metadata, cached_ids, sampling_output):
        output_len_count = plugin_object.input_manager.cache.output_len_count[cached_ids]
        for i, token_id in enumerate(sampling_output.token_ids):
            _cur_out_len = output_len_count[i]
            if input_metadata.batch_request_ids[i] not in ServiceField.req_id_and_max_decode_length:
                continue
            _max_out_len = ServiceField.req_id_and_max_decode_length[input_metadata.batch_request_ids[i]]
            if _cur_out_len < (_max_out_len - 1):
                if token_id == plugin_object.eos_token_id:
                    token_id = np.random.randint(0, plugin_object.model_wrapper.config.vocab_size)
                if sampling_output.top_token_ids.size == 0:
                    continue
                if i >= sampling_output.top_token_ids.size:
                    continue

                if sampling_output.top_token_ids[i].any() and plugin_object.eos_token_id in \
                        sampling_output.top_token_ids[i]:
                    sampling_output.top_token_ids[i] = np.where(
                        sampling_output.top_token_ids[i] == plugin_object.eos_token_id,
                        np.random.randint(0, plugin_object.model_wrapper.config.vocab_size),
                        sampling_output.top_token_ids[i])
            else:
                sampling_output.token_ids[i] = plugin_object.eos_token_id
                if sampling_output.top_token_ids.size == 0:
                    continue
                if i >= sampling_output.top_token_ids.size:
                    continue
                if sampling_output.top_token_ids[i].any():
                    sampling_output.top_token_ids[i] = np.where(
                        sampling_output.top_token_ids[i] != plugin_object.eos_token_id,
                        plugin_object.eos_token_id,
                        sampling_output.top_token_ids[i])


    @staticmethod
    def predict(time_sleep: bool = True):
        """

        return: time ms
        """
        time_sleep = os.getenv(IS_SLEEP_FLAG, str(time_sleep)).lower().strip() == "true"
        # 增加缓存
        st = time.perf_counter()
        _cache_key = (ServiceField.batch_field, ServiceField.request_field)
        if _cache_key not in Simulate.predict_cache:
            predict_res = predict_v1_with_cache(ServiceField.batch_field, ServiceField.request_field,
                                                ServiceField.config_path, ServiceField.fh, ServiceField.data_processor,
                                                cache_data=ServiceField.config_path.cache_data)
            Simulate.predict_cache[_cache_key] = predict_res
        else:
            predict_res = Simulate.predict_cache.get(_cache_key)

        for _pre_v in predict_res:
            if _pre_v == -1:
                continue
            if time_sleep:
                _run_time = time.perf_counter() - st
                _wait_time = _pre_v / 10 ** 6 - _run_time
                if _wait_time > 0:
                    time.sleep(_wait_time)
                return _pre_v
            else:
                return _pre_v

    @staticmethod
    def predict_and_save(time_sleep: bool = False):
        res = Simulate.predict(time_sleep)
        # 增加异步写入能力
        try:
            predict_queue.put_nowait(res)
        except queue.Full:
            logger.error("predict_queue is full")
            predict_queue.put(res)