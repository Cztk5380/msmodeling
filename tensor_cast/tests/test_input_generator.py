import unittest

import torch
from parameterized import parameterized

from ..core.input_generator import generate_inputs, generate_inputs_varlen, RequestInfo

from ..device import TEST_DEVICE
from ..layers.attention import AttentionTensorCast
from ..model_config import ModelConfig, ParallelConfig, QuantConfig
from ..performance_model.analytic import AnalyticPerformanceModel
from ..runtime import Runtime
from ..transformers.model import TransformerModel
from ..transformers.utils import AutoModelConfigLoader


class InputGeneratorTestCase(unittest.TestCase):
    @parameterized.expand([[True], [False]])
    def test_selected_token_indices_for_lmhead(self, is_decode):
        auto_loader = AutoModelConfigLoader()
        model_id = "Qwen/Qwen3-32B"
        hf_config = auto_loader.load_config(model_id)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            enable_repetition=True,
            hf_config=hf_config,
        )
        model = TransformerModel(model_id, model_config)

        query_len = 100
        batch_size = 2
        inputs = generate_inputs(
            model,
            [
                RequestInfo(
                    query_len=query_len,
                    seq_len=query_len,
                    concurrency=batch_size,
                    is_decode=is_decode,
                )
            ],
        )
        if is_decode:
            output_shape = (1, batch_size * query_len, model.vocab_size)
        else:
            output_shape = (1, batch_size, model.vocab_size)

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config), torch.no_grad():
            outputs = model.forward(**inputs)
            self.assertEqual(outputs.shape, output_shape)

    @parameterized.expand([[True], [False]])
    def test_varlen_selected_token_indices_for_lmhead(self, is_decode):
        auto_loader = AutoModelConfigLoader()
        model_id = "Qwen/Qwen3-32B"
        hf_config = auto_loader.load_config(model_id)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            attention_cls=AttentionTensorCast,
            enable_repetition=True,
            hf_config=hf_config,
        )
        model = TransformerModel(model_id, model_config)

        query_len = [90, 110]
        batch_size = len(query_len)
        request_infos = []
        for i in range(batch_size):
            request_infos.append(
                RequestInfo(
                    query_len=query_len[i],
                    seq_len=query_len[i],
                    is_decode=is_decode,
                )
            )
        inputs = generate_inputs_varlen(model, request_infos, 128)
        if is_decode:
            output_shape = (1, sum(query_len), model.vocab_size)
        else:
            output_shape = (1, batch_size, model.vocab_size)

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config), torch.no_grad():
            outputs = model.forward(**inputs)
            self.assertEqual(outputs.shape, output_shape)
