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
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from experimental.inference.constant import BatchStage
from experimental.inference.simulate import ServiceField
from experimental.inference.simulate_vllm import SimulateVllm


class TestGenerateFeatures:
    @staticmethod
    def test_generate_features_no_input():
        class ModelInput:
            is_prompt = None
            seq_lens = None

        batch_field, request_field = SimulateVllm.generate_features(ModelInput())
        assert batch_field is None
        assert request_field is None

    @staticmethod
    def test_generate_features_prompt_no_block_tables():
        class ModelInput:
            is_prompt = True
            seq_lens = [1, 2, 3]
            attn_metadata = type('', (), {})()
            attn_metadata.block_tables = None
            request_ids_to_seq_ids = {"1": [1], "2": [2], "3": [3]}
            sampling_metadata = type('', (), {})()
            sampling_metadata.seq_groups = [
                type('', (), {})(),
                type('', (), {})(),
                type('', (), {})()
            ]
            sampling_metadata.seq_groups[0].seq_ids = [1]
            sampling_metadata.seq_groups[0].sampling_params = type('', (), {"stop_token_ids": None})()
            sampling_metadata.seq_groups[0].seq_data = {1: type('', (), {})()}
            sampling_metadata.seq_groups[0].seq_data[1].prompt_token_ids = np.array([1, 2, 3])
            sampling_metadata.seq_groups[0].seq_data[1].output_token_ids = np.array([4, 5, 6])
            sampling_metadata.seq_groups[1].seq_ids = [2]
            sampling_metadata.seq_groups[1].sampling_params = type('', (), {"stop_token_ids": None})()
            sampling_metadata.seq_groups[1].seq_data = {2: type('', (), {})()}
            sampling_metadata.seq_groups[1].seq_data[2].prompt_token_ids = np.array([7, 8, 9])
            sampling_metadata.seq_groups[1].seq_data[2].output_token_ids = np.array([10, 11, 12])
            sampling_metadata.seq_groups[2].seq_ids = [3]
            sampling_metadata.seq_groups[2].sampling_params = type('', (), {"stop_token_ids": None})()
            sampling_metadata.seq_groups[2].seq_data = {3: type('', (), {})()}
            sampling_metadata.seq_groups[2].seq_data[3].prompt_token_ids = np.array([13, 14, 15])
            sampling_metadata.seq_groups[2].seq_data[3].output_token_ids = np.array([16, 17, 18])

        batch_field, request_field = SimulateVllm.generate_features(ModelInput())
        assert batch_field.batch_stage == BatchStage.PREFILL
        assert batch_field.batch_size == 3
        assert batch_field.total_prefill_token == 9
        assert batch_field.total_need_blocks == 0
        assert batch_field.max_seq_len == 3
        assert len(request_field) == 3
        assert request_field[0].input_length == 3
        assert request_field[0].need_blocks == 0
        assert request_field[0].output_length == 3
        assert request_field[1].input_length == 3
        assert request_field[1].need_blocks == 0
        assert request_field[1].output_length == 3
        assert request_field[2].input_length == 3
        assert request_field[2].need_blocks == 0
        assert request_field[2].output_length == 3

    @staticmethod
    def test_generate_features_decode_with_block_tables():
        class ModelInput:
            is_prompt = False
            seq_lens = [1, 2, 3]
            attn_metadata = type('', (), {})()
            attn_metadata.block_tables = torch.tensor([[-1, -1, -1], [0, 1, 2], [3, 4, 5]])
            request_ids_to_seq_ids = {"1": [1], "2": [2], "3": [3]}
            sampling_metadata = type('', (), {})()
            sampling_metadata.seq_groups = [
                type('', (), {})(),
                type('', (), {})(),
                type('', (), {})()
            ]
            sampling_metadata.seq_groups[0].seq_ids = [1]
            sampling_metadata.seq_groups[0].sampling_params = type('', (), {"stop_token_ids": None})()
            sampling_metadata.seq_groups[0].seq_data = {1: type('', (), {})()}
            sampling_metadata.seq_groups[0].seq_data[1].prompt_token_ids = np.array([1, 2, 3])
            sampling_metadata.seq_groups[0].seq_data[1].output_token_ids = np.array([4, 5, 6])
            sampling_metadata.seq_groups[1].seq_ids = [2]
            sampling_metadata.seq_groups[1].sampling_params = type('', (), {"stop_token_ids": None})()
            sampling_metadata.seq_groups[1].seq_data = {2: type('', (), {})()}
            sampling_metadata.seq_groups[1].seq_data[2].prompt_token_ids = np.array([7, 8, 9])
            sampling_metadata.seq_groups[1].seq_data[2].output_token_ids = np.array([10, 11, 12])
            sampling_metadata.seq_groups[2].seq_ids = [3]
            sampling_metadata.seq_groups[2].sampling_params = type('', (), {"stop_token_ids": None})()
            sampling_metadata.seq_groups[2].seq_data = {3: type('', (), {})()}
            sampling_metadata.seq_groups[2].seq_data[3].prompt_token_ids = np.array([13, 14, 15])
            sampling_metadata.seq_groups[2].seq_data[3].output_token_ids = np.array([16, 17, 18])

        batch_field, request_field = SimulateVllm.generate_features(ModelInput())
        assert batch_field.batch_stage == BatchStage.DECODE
        assert batch_field.batch_size == 3
        assert batch_field.total_need_blocks == 6
        assert batch_field.total_prefill_token == 9
        assert batch_field.max_seq_len == 3
        assert len(request_field) == 3
        assert request_field[0].input_length == 3
        assert request_field[0].need_blocks == 0
        assert request_field[0].output_length == 3
        assert request_field[1].input_length == 3
        assert request_field[1].need_blocks == 3
        assert request_field[1].output_length == 3
        assert request_field[2].input_length == 3
        assert request_field[2].need_blocks == 3
        assert request_field[2].output_length == 3


# 在每个测试用例之前运行，确保环境是干净的
class TestMaxOutputLen:
    @staticmethod
    def test_get_max_output_len_from_json():
        # 模拟json文件中的数据
        ServiceField.req_id_and_max_decode_length = {0: 100, 1: 200, 2: 300}
        assert SimulateVllm.get_max_output_len(0) == 100
        assert SimulateVllm.get_max_output_len(1) == 200
        assert SimulateVllm.get_max_output_len(2) == 300

    @staticmethod
    def test_get_max_output_len_from_int_conversion():
        # 模拟json文件中的数据
        ServiceField.req_id_and_max_decode_length = {0: 100, 1: 200, 2: 300}
        assert SimulateVllm.get_max_output_len('0') == 100
        assert SimulateVllm.get_max_output_len('1') == 200
        assert SimulateVllm.get_max_output_len('2') == 300

    @staticmethod
    def test_get_max_output_len_from_cache():
        # 模拟缓存中的数据
        SimulateVllm.req_id_to_max_token_by_sequence = {'cmpl-7d6e773db843411985fba579778e81ea-0': 400}
        assert SimulateVllm.get_max_output_len('cmpl-7d6e773db843411985fba579778e81ea-0') == 400

    @staticmethod
    def test_get_max_output_len_from_sequence_length():
        # 模拟序列长度的数据
        ServiceField.req_id_and_max_decode_length = {0: 100, 1: 200, 2: 300}
        assert SimulateVllm.get_max_output_len('cmpl-7d6e773db843411985fba579778e81ea-0') == 100
        assert SimulateVllm.get_max_output_len('cmpl-7d6e773db843411985fba579778e81ea-1') == 200
        assert SimulateVllm.get_max_output_len('cmpl-7d6e773db843411985fba579778e81ea-2') == 300

    @staticmethod
    def test_get_max_output_len_from_sequence_length_str():
        # 模拟序列长度的数据
        ServiceField.req_id_and_max_decode_length = {'0': 100, '1': 200, '2': 300}
        assert SimulateVllm.get_max_output_len('cmpl-7d6e773db843411985fba579778e81ea-0') == 100
        assert SimulateVllm.get_max_output_len('cmpl-7d6e773db843411985fba579778e81ea-1') == 200
        assert SimulateVllm.get_max_output_len('cmpl-7d6e773db843411985fba579778e81ea-2') == 300

    @pytest.fixture(autouse=True)
    def setup(self):
        ServiceField.req_id_and_max_decode_length = {}
        SimulateVllm.req_id_to_max_token_by_sequence = {}


class TestCurOutputLen:
    # 测试用例1: 当req_id在req_to_output_len字典中找到时
    @staticmethod
    def test_get_cur_output_len_found():
        SimulateVllm.req_to_output_len = {1: 10, '2': 20}
        assert SimulateVllm.get_cur_output_len(1) == 10
        assert SimulateVllm.get_cur_output_len('2') == 20

    # 测试用例2: 当req_id不在req_to_output_len字典中，但可以转换为整数并在字典中找到时
    @staticmethod
    def test_get_cur_output_len_found_after_conversion():
        SimulateVllm.req_to_output_len = {1: 10, '2': 20}
        assert SimulateVllm.get_cur_output_len('1') == 10
        assert SimulateVllm.get_cur_output_len('2') == 20

    # 测试用例3: 当req_id不在req_to_output_len字典中，且无法转换为整数时
    @staticmethod
    def test_get_cur_output_len_not_found():
        SimulateVllm.req_to_output_len = {1: 10, '2': 20}
        assert SimulateVllm.get_cur_output_len('3') is None
        assert SimulateVllm.get_cur_output_len(3) is None
        assert SimulateVllm.get_cur_output_len('abc') is None


class TestSimulateVllmUpdateToken:
    @staticmethod
    def test_update_token_with_cur_out_len_less_than_max_out_len(setup):
        model_input, sampling_output = setup

        # Mocking static method return values
        SimulateVllm.get_max_output_len.return_value = 5
        SimulateVllm.get_cur_output_len.return_value = 3

        SimulateVllm.req_to_stop_token_ids = {'req1': [151645]}

        SimulateVllm.update_token(model_input, sampling_output)

        assert sampling_output.outputs[0].samples[0].output_token == 46310
        assert sampling_output.outputs[1].samples[0].output_token == 38107
        assert sampling_output.outputs[2].samples[0].output_token == 88089

    @staticmethod
    def test_update_token_with_cur_out_len_greater_than_max_out_len_minus_one(setup):
        model_input, sampling_output = setup

        # Mocking static method return values
        SimulateVllm.get_max_output_len.return_value = 5
        SimulateVllm.get_cur_output_len.return_value = 5

        SimulateVllm.req_to_stop_token_ids = {'req1': [151645]}

        SimulateVllm.update_token(model_input, sampling_output)

        assert sampling_output.outputs[0].samples[0].output_token == 151645
        assert sampling_output.outputs[1].samples[0].output_token == 151645
        assert sampling_output.outputs[2].samples[0].output_token == 151645
        assert 46310 not in sampling_output.outputs[0].samples[0].logprobs
        assert 38107 not in sampling_output.outputs[1].samples[0].logprobs
        assert 88089 not in sampling_output.outputs[2].samples[0].logprobs

    @staticmethod
    def test_update_token_with_none_max_out_len(setup):
        model_input, sampling_output = setup

        # Mocking static method return values
        SimulateVllm.get_max_output_len.return_value = None
        SimulateVllm.get_cur_output_len.return_value = 3

        SimulateVllm.update_token(model_input, sampling_output)

        assert sampling_output.outputs[0].samples[0].output_token == 46310
        assert sampling_output.outputs[1].samples[0].output_token == 38107
        assert sampling_output.outputs[2].samples[0].output_token == 88089

    @staticmethod
    def test_update_token_with_none_cur_out_len(setup):
        model_input, sampling_output = setup

        # Mocking static method return values
        SimulateVllm.get_max_output_len.return_value = 5
        SimulateVllm.get_cur_output_len.return_value = None

        SimulateVllm.update_token(model_input, sampling_output)

        assert sampling_output.outputs[0].samples[0].output_token == 46310
        assert sampling_output.outputs[1].samples[0].output_token == 38107
        assert sampling_output.outputs[2].samples[0].output_token == 88089

    @pytest.fixture(autouse=True)
    def setup(self):
        # Setup mock objects
        # Mocking static methods
        SimulateVllm.get_max_output_len = MagicMock()
        SimulateVllm.get_cur_output_len = MagicMock()

        # Mocking static attributes
        SimulateVllm.req_to_stop_token_ids = defaultdict(list)
        model_input = MagicMock()
        model_input.request_ids_to_seq_ids = {'0': [0], '1': [1], '2': [2], '3': [3]}
        sampling_output = MagicMock()
        sampling_output.outputs = [type('CompletionSequenceGroupOutput', (), {})(),
                                   type('CompletionSequenceGroupOutput', (), {})(),
                                   type('CompletionSequenceGroupOutput', (), {})()]
        sampling_output.outputs[0].samples = [type('SequenceOutput', (), {})()]
        sampling_output.outputs[0].samples[0].output_token = 46310
        sampling_output.outputs[0].samples[0].parent_seq_id = 0
        sampling_output.outputs[0].samples[0].logprobs = {46310: type('Logprob', (), {})()}
        sampling_output.outputs[1].samples = [type('SequenceOutput', (), {})()]
        sampling_output.outputs[1].samples[0].output_token = 38107
        sampling_output.outputs[1].samples[0].parent_seq_id = 1
        sampling_output.outputs[1].samples[0].logprobs = {38107: type('Logprob', (), {})()}
        sampling_output.outputs[2].samples = [type('SequenceOutput', (), {})()]
        sampling_output.outputs[2].samples[0].output_token = 88089
        sampling_output.outputs[2].samples[0].parent_seq_id = 2
        sampling_output.outputs[2].samples[0].logprobs = {88089: type('Logprob', (), {})()}
        return model_input, sampling_output
