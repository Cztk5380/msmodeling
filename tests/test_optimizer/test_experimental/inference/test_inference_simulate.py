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
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from src.optimize.experimental.config.config import get_settings
from src.optimize.experimental.inference.constant import IS_SLEEP_FLAG
from src.optimize.experimental.inference.data_format_v1 import BatchField, RequestField
from src.optimize.experimental.inference.simulate import Simulate, predict_queue, ServiceField, FileLogger


class TestFileLogger:
    @staticmethod
    def test_open_file_with_path(logger, file_path):
        logger.file_path = file_path
        logger.mode = 'w'
        logger.open_file()
        assert logger.fout is not None
        assert logger.fout.closed is False
        logger.fout.close()

    @staticmethod
    def test_open_file_with_string(logger, file_path):
        logger.file_path = str(file_path)
        logger.mode = 'w'
        logger.open_file()
        assert logger.fout is not None
        assert logger.fout.closed is False
        logger.fout.close()

    @staticmethod
    def test_open_file_with_invalid_path(logger):
        logger.file_path = None
        logger.mode = 'w'
        with pytest.raises(TypeError):
            logger.open_file()

    @staticmethod
    def test_open_file_with_invalid_mode(logger, file_path):
        logger.file_path = file_path
        logger.mode = 'x'
        logger.open_file()

    @staticmethod
    def test_write_with_none_fout(logger, file_path):
        logger.fout = None
        logger.lock = threading.Lock()
        message = "test message"
        logger.write(message)
        # Since fout is None, no exception should be raised and nothing to assert

    @staticmethod
    def test_write_with_not_none_fout(logger, file_path):
        mock_file = MagicMock()
        logger.fout = mock_file
        logger.lock = threading.Lock()
        message = "test message"
        logger.write(message)
        # Check if write and flush are called correctly
        mock_file.write.assert_any_call(message)
        mock_file.write.assert_any_call("\n")
        mock_file.flush.assert_called_once()

    @pytest.fixture
    def logger(self):
        return FileLogger(Path(get_settings().simulator_output).joinpath(f"simulate_{os.getpid()}.csv"))
    
    @pytest.fixture
    def file_path(self):
        return Path("test.log")


class TestSimulate:
    @staticmethod
    def test_generate_random_token_shape(plugin_object):
        shape = (2, 3)
        max_value = 32000
        result = Simulate.generate_random_token(plugin_object, shape, max_value)
        assert result.shape == shape, "Generated array shape does not match the expected shape."

    @staticmethod
    def test_generate_random_token_eos_token_replacement(plugin_object):
        shape = (2, 3)
        max_value = 32000
        result = Simulate.generate_random_token(plugin_object, shape, max_value)
        assert plugin_object.eos_token_id not in result, "eos_token_id should be replaced in the generated array."

    @staticmethod
    def test_generate_random_token_value_range(plugin_object):
        shape = (2, 3)
        max_value = 32000
        result = Simulate.generate_random_token(plugin_object, shape, max_value)
        assert result.min() >= 0 and result.max() <= max_value, "Generated array values are out of the expected range."

    @staticmethod
    def test_generate_random_token_no_replacement_needed(plugin_object):
        shape = (2, 3)
        max_value = 32000
        result = Simulate.generate_random_token(plugin_object, shape, max_value)
        assert result.size == np.prod(shape), "The size of the generated array does not match the product of the shape."

    @pytest.fixture
    def plugin_object(self):
        class PluginObject:
            def __init__(self):
                self.eos_token_id = 10000  # 假设的eos_token_id

        return PluginObject()


def test_generate_logits():
    device = "cpu"
    # 测试不同的batch_size和vocab_size
    for batch_size in [1, 2, 10]:
        for vocab_size in [128, 1024, 129280]:
            logits = Simulate.generate_logits(batch_size, vocab_size, device=device)
            assert logits.shape == (
            batch_size, vocab_size), f"Expected shape ({batch_size}, {vocab_size}), got {logits.shape}"

    # 测试不同的dtype
    for dtype in ["float16", "bfloat16", "float"]:
        logits = Simulate.generate_logits(1, device=device, dtype=dtype)
        if dtype == "float16":
            assert logits.dtype == torch.float16, f"Expected dtype torch.float16, got {logits.dtype}"
        elif dtype == "bfloat16":
            assert logits.dtype == torch.bfloat16, f"Expected dtype torch.bfloat16, got {logits.dtype}"
        elif dtype == "float":
            assert logits.dtype == torch.float, f"Expected dtype torch.float, got {logits.dtype}"


class TestSimulateUpdateToken:
    @staticmethod
    def test_update_token_with_eos_token(setup):
        plugin_object, input_metadata, cached_ids, sampling_output = setup
        ServiceField.req_id_and_max_decode_length = {0: 10}

        Simulate.update_token(plugin_object, input_metadata, cached_ids, sampling_output)

        assert sampling_output.token_ids[0].item() == plugin_object.eos_token_id
        assert sampling_output.top_token_ids.size == 0

    @staticmethod
    def test_update_token_with_top_token(setup):
        plugin_object, input_metadata, cached_ids, sampling_output = setup
        ServiceField.req_id_and_max_decode_length = {0: 10}
        sampling_output.top_token_ids = np.array([[50256]])
        Simulate.update_token(plugin_object, input_metadata, cached_ids, sampling_output)

        assert sampling_output.token_ids[0].item() == plugin_object.eos_token_id
        assert sampling_output.top_token_ids[0].item() != plugin_object.eos_token_id

    @staticmethod
    def test_update_token_with_max_length_reached(setup):
        plugin_object, input_metadata, cached_ids, sampling_output = setup
        ServiceField.req_id_and_max_decode_length = {0: 1}
        sampling_output.token_ids = np.array([50224])
        Simulate.update_token(plugin_object, input_metadata, cached_ids, sampling_output)

        assert sampling_output.token_ids[0] == plugin_object.eos_token_id
        assert sampling_output.top_token_ids.size == 0

    @staticmethod
    def test_update_token_with_max_length_reached_with_top_token(setup):
        plugin_object, input_metadata, cached_ids, sampling_output = setup
        ServiceField.req_id_and_max_decode_length = {0: 1}
        sampling_output.token_ids = np.array([50224])
        sampling_output.top_token_ids = np.array([[50224]])
        Simulate.update_token(plugin_object, input_metadata, cached_ids, sampling_output)

        assert sampling_output.token_ids[0] == plugin_object.eos_token_id
        assert sampling_output.top_token_ids[0].item() == plugin_object.eos_token_id

    @staticmethod
    def test_update_token_with_no_request_id(setup):
        plugin_object, input_metadata, cached_ids, sampling_output = setup
        ServiceField.req_id_and_max_decode_length = {}

        Simulate.update_token(plugin_object, input_metadata, cached_ids, sampling_output)

        # 由于没有请求ID，方法应该不做任何更改
        assert sampling_output.token_ids[0] == plugin_object.eos_token_id
        assert sampling_output.top_token_ids.size == 0

    @pytest.fixture
    def setup(self):
        # 创建模拟对象
        plugin_object = MagicMock()
        plugin_object.input_manager.cache.output_len_count = np.full((3, 1), 1, dtype=np.int32)
        plugin_object.eos_token_id = 50256
        plugin_object.model_wrapper.config.vocab_size = 50256

        input_metadata = MagicMock()
        input_metadata.batch_request_ids = np.array([0])

        cached_ids = 0

        sampling_output = MagicMock()
        sampling_output.token_ids = np.array([50256])
        sampling_output.top_token_ids = np.zeros((1, 0), dtype=np.int32)

        return plugin_object, input_metadata, cached_ids, sampling_output


class MockFileHandler:
    def __init__(self):
        pass


class MockDataProcessor:
    def __init__(self):
        pass


class MockConfigPath:
    def __init__(self):
        self.cache_data = {}


@pytest.fixture
def predict_setup():
    Simulate.predict_cache = {}
    ServiceField.batch_field = BatchField("decode", 20, 20.0, 580.0, 29.0)
    ServiceField.request_field = (
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
        RequestField(29.0, 1, 2),
    )
    ServiceField.config_path = MockConfigPath()
    ServiceField.fh = MockFileHandler()
    ServiceField.data_processor = MockDataProcessor()


# Test cases
def test_predict_with_sleep(predict_setup, monkeypatch):
    monkeypatch.setattr("src.optimize.experimental.inference.simulate.predict_v1_with_cache", \
                        lambda *args, **kwargs: (-1, 300000))
    assert len(Simulate.predict_cache) == 0
    st = time.perf_counter()
    os.environ[IS_SLEEP_FLAG] = "true"
    result = Simulate.predict()
    assert time.perf_counter() - st > 0.3
    assert len(Simulate.predict_cache) == 1
    assert result == 300000
    result = Simulate.predict()
    assert result == 300000
    assert len(Simulate.predict_cache) == 1


def test_predict_without_sleep(predict_setup, monkeypatch):
    monkeypatch.setattr("src.optimize.experimental.inference.simulate.predict_v1_with_cache", \
                        lambda *args, **kwargs: (-1, 300000))
    os.environ[IS_SLEEP_FLAG] = "false"
    assert len(Simulate.predict_cache) == 0
    st = time.perf_counter()
    result = Simulate.predict()
    assert time.perf_counter() - st < 0.3
    assert result == 300000
    assert len(Simulate.predict_cache) == 1
    result = Simulate.predict()
    assert result == 300000
    assert len(Simulate.predict_cache) == 1


@patch('src.optimize.experimental.inference.simulate.Simulate.predict')
def test_predict_and_save(mock_predict):
    # 测试predict_and_save方法
    # 模拟predict方法的返回值
    mock_predict.return_value = MagicMock()

    # 调用predict_and_save方法
    Simulate.predict_and_save()

    # 确保predict方法被调用
    mock_predict.assert_called_once()

    # 确保predict_queue中有一个元素
    assert predict_queue.qsize() == 1

    # 清空predict_queue
    while not predict_queue.empty():
        predict_queue.get()

    # 测试predict_and_save方法，time_sleep为True
    # 调用predict_and_save方法
    Simulate.predict_and_save(time_sleep=True)

    # 确保predict方法被调用
    assert mock_predict.call_count == 2

    # 确保predict_queue中有一个元素
    assert predict_queue.qsize() == 1

    # 清空predict_queue
    while not predict_queue.empty():
        predict_queue.get()
