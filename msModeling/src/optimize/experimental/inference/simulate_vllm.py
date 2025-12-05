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

import json
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import msserviceprofiler.modelevalstate.inference.simulate as simulate
from msserviceprofiler.modelevalstate.inference.constant import BatchStage
from msserviceprofiler.modelevalstate.inference.data_format_v1 import BatchField, RequestField
from msserviceprofiler.modelevalstate.inference.dataset import CustomLabelEncoder, preset_category_data, DataProcessor
from msserviceprofiler.modelevalstate.inference.file_reader import FileHanlder, StaticFile
from msserviceprofiler.modelevalstate.inference.simulate import ServiceField, write_file, file_log
from msserviceprofiler.msguard.security import open_s


class SimulateVllm:
    first = True
    predict_cache = {}
    req_to_output_len = defaultdict(int)
    req_to_stop_token_ids = {}
    req_id_to_max_token_by_sequence = {}
 
    @staticmethod
    def init():
        if SimulateVllm.first:
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
            elif not Path(ServiceField.config_path.static_file_dir).is_dir():
                logger.error(f"{ServiceField.config_path.static_file_dir!r} is not a directory.")
            static_file = StaticFile(base_path=ServiceField.config_path.static_file_dir)
            ServiceField.fh = FileHanlder(static_file)
            ServiceField.fh.load_static_data()
            custom_encoder = CustomLabelEncoder(preset_category_data)
            custom_encoder.fit()
            ServiceField.data_processor = DataProcessor(custom_encoder)
            SimulateVllm.first = False
            simulate.sub_thread = threading.Thread(target=write_file, args=(file_log,))
            simulate.sub_thread.start()
    
    @staticmethod
    def process_batch(model_input):
        if model_input.is_prompt:
            batch_stage = BatchStage.PREFILL
        else:
            batch_stage = BatchStage.DECODE
        if model_input.attn_metadata.block_tables is None:
            all_need_blocks = torch.zeros((len(model_input.seq_lens), 1))
        else:
            all_need_blocks = torch.count_nonzero(model_input.attn_metadata.block_tables > -1, dim=-1)
        return batch_stage, all_need_blocks
    
    @staticmethod
    def generate_features(model_input):
        if model_input.is_prompt is None and model_input.seq_lens is None:
            return None, None
        batch_stage, all_need_blocks = SimulateVllm.process_batch(model_input)
        _total_req_input_len = []
        _all_request_ids = set({})
        _req_to_input_len = defaultdict(int)
        SimulateVllm.req_to_output_len = defaultdict(int)
        _req_to_need_blocks = defaultdict(int)
        _seq_id_to_req = {}
        for _req_id, _seq_ids in model_input.request_ids_to_seq_ids.items():
            for _id in _seq_ids:
                _seq_id_to_req[_id] = _req_id
        for _index, _seq_group_sample in enumerate(model_input.sampling_metadata.seq_groups):
            _stop_ids = _seq_group_sample.sampling_params.stop_token_ids
            for _seq_id in _seq_group_sample.seq_ids:
                if _seq_id not in _seq_group_sample.seq_data:
                    logger.warning(f"{_seq_id} not in {_seq_group_sample.seq_data}")
                    continue
                _cur_req_id = _seq_id_to_req.get(_seq_id)
                if _cur_req_id is None:
                    raise ValueError(f"No request ID found for sequence ID {_seq_id}.")
                if _stop_ids:
                    SimulateVllm.req_to_stop_token_ids[_cur_req_id] = _stop_ids
                _seq_data = _seq_group_sample.seq_data.get(_seq_id)
                if _seq_data is not None:
                    _req_input_len = len(_seq_data.prompt_token_ids)
                else:
                    _req_input_len = 0
                _req_to_input_len[_cur_req_id] += _req_input_len
                _total_req_input_len.append(_req_input_len)
                _seq_data = _seq_group_sample.seq_data.get(_seq_id)
                if _seq_data is not None and hasattr(_seq_data, 'output_token_ids'):
                    _req_output_len = len(_seq_data.output_token_ids)
                else:
                    _req_output_len = 0
                SimulateVllm.req_to_output_len[_cur_req_id] += _req_output_len
                _req_need_block = all_need_blocks[_index].sum().item()
                _req_to_need_blocks[_cur_req_id] += _req_need_block
                _all_request_ids.add(_cur_req_id)
        request_field = []
        for _cur_id in _all_request_ids:
            request_field.append(RequestField(_req_to_input_len.get(_cur_id), _req_to_need_blocks.get(_cur_id),
                                              SimulateVllm.req_to_output_len.get(_cur_id)))
        batch_field = BatchField(batch_stage, len(model_input.seq_lens), all_need_blocks.sum().item(),
                                 sum(_total_req_input_len), max(_total_req_input_len))
        ServiceField.batch_field = batch_field
        ServiceField.request_field = tuple(sorted(request_field))
        return batch_field, tuple(sorted(request_field))

    @staticmethod
    def get_max_output_len(req_id):
        """
        req_id: 离线情况下，会是数字0,1,2， 在线模型会是字符串 cmpl-7d6e773db843411985fba579778e81ea-0
        """
        # 尝试直接从json文件中获取
        _max_out_len = ServiceField.req_id_and_max_decode_length.get(req_id)
        if _max_out_len is None:
            try:
                # 尝试转为int类型进行获取
                _max_out_len = ServiceField.req_id_and_max_decode_length[int(req_id)]
            except ValueError:
                pass
        # 尝试从缓存中获取
        if _max_out_len is None:
            _max_out_len = SimulateVllm.req_id_to_max_token_by_sequence.get(req_id)
        # 按照收到请求的序号，从采集收集的json文件里面获取对应序号的数据。
        if _max_out_len is None:
            _max_out_len = ServiceField.req_id_and_max_decode_length.get(
                len(SimulateVllm.req_id_to_max_token_by_sequence))
            if _max_out_len:
                SimulateVllm.req_id_to_max_token_by_sequence[req_id] = _max_out_len
        if _max_out_len is None:
            _max_out_len = ServiceField.req_id_and_max_decode_length.get(
                str(len(SimulateVllm.req_id_to_max_token_by_sequence)))
            if _max_out_len:
                SimulateVllm.req_id_to_max_token_by_sequence[req_id] = _max_out_len
        return _max_out_len

    @staticmethod
    def get_cur_output_len(req_id):
        _cur_out_len = SimulateVllm.req_to_output_len.get(req_id)
        if _cur_out_len is None:
            try:
                _cur_out_len = SimulateVllm.req_to_output_len.get(int(req_id))
            except ValueError:
                pass
        return _cur_out_len
 
    @staticmethod
    def update_token(model_input, sampling_output, eos_token_id=151645):
        # sampling output 是vllm.model_executor.layers.sampler.SamplerOutput 类
        for _com_seq_gp in sampling_output.outputs:
            # 每一条_com_seq_gp 对应的输出 _seq_out, _com_seq_gp 对应输入的index的输出。
            for _seq_out in _com_seq_gp.samples:
                # 当前output的对应的输入seq id。
                _p_s_id = _seq_out.parent_seq_id
                # 当前request id 可能会有多个seq，将所有seq的输出累加到一起作为这个request的输出。
                for _req_id, _seq_ids in model_input.request_ids_to_seq_ids.items():
                    # 当前输出是否不属于这个请求的，不处理。
                    if _p_s_id not in _seq_ids:
                        continue
                    _max_out_len = SimulateVllm.get_max_output_len(_req_id)
                    # 未找到采集时该请求的最大输出次数，不处理。
                    if _max_out_len is None:
                        continue
                    # 获取该请求当前
                    _cur_out_len = SimulateVllm.get_cur_output_len(_req_id)
                    if _cur_out_len is None:
                        continue
                    # 更新这个SequenceOutput的输出token。
                    _origin_token = _seq_out.output_token
                    _cur_req_eos_token_id = SimulateVllm.req_to_stop_token_ids.get(_req_id, [eos_token_id])
                    if _cur_out_len < (_max_out_len - 1) and _seq_out.output_token in _cur_req_eos_token_id:
                        _seq_out.output_token = np.random.randint(0, min(_cur_req_eos_token_id))
                        _seq_out.logprobs[_seq_out.output_token] = _seq_out.logprobs.get(_origin_token)
                        _seq_out.logprobs.pop(_origin_token)
                    elif _cur_out_len >= (_max_out_len - 1):
                        _seq_out.output_token = _cur_req_eos_token_id[0]
                        _seq_out.logprobs[_seq_out.output_token] = _seq_out.logprobs.get(_origin_token)
                        _seq_out.logprobs.pop(_origin_token)
                    # 处理完这个请求
                    break
