import unittest
from unittest.mock import Mock

import torch
from parameterized import parameterized

from ..core.input_generator import (
    _get_padding_alignment,
    generate_inputs,
    generate_inputs_varlen,
    RequestInfo,
)

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

    # Define test cases: (scenario description, moe_tensor_size, tensor_size, has_ep, num_experts, expected result)
    test_cases = [
        (
            "moe_tensor_parallel_size not equal and has_ep is True",
            2,
            4,
            True,
            8,
            32,  # 8*4=32
        ),
        (
            "moe_tensor_parallel_size equal and has_ep is True",
            4,
            4,
            True,
            None,
            4,  # Go to else branch, return tensor_size
        ),
        (
            "moe_tensor_parallel_size not equal and has_ep is False",
            2,
            4,
            False,
            None,
            4,  # Go to else branch, return tensor_size
        ),
        (
            "moe_tensor_parallel_size equal and has_ep is False",
            8,
            8,
            False,
            None,
            8,  # Go to else branch, return tensor_size
        ),
        (
            "Edge case: tensor_parallel_size is 0",
            0,
            0,
            False,
            None,
            0,  # Boundary value test
        ),
    ]

    @parameterized.expand(test_cases)
    def test_get_padding_alignment(
        self, desc, moe_tensor_size, tensor_size, has_ep, num_experts, expected
    ):
        """Parameterized test for all scenarios of _get_padding_alignment function"""
        # 1. Mock parallel_config object
        parallel_config = Mock()
        parallel_config.moe_tensor_parallel_size = moe_tensor_size
        parallel_config.tensor_parallel_size = tensor_size
        parallel_config.has_ep = Mock(return_value=has_ep)

        # 2. Mock hf_config object (set num_experts only when needed)
        hf_config = Mock()
        if num_experts is not None:
            hf_config.num_experts = num_experts
        moe_config = Mock()
        moe_config.num_experts_key = "num_experts"

        # 3. Mock model_config object
        model_config = Mock()
        model_config.parallel_config = parallel_config
        model_config.hf_config = hf_config
        model_config.moe_config = moe_config

        # 4. Execute function and assert result
        result = _get_padding_alignment(model_config)
        assert result == expected, (
            f"Scenario [{desc}] test failed: expected {expected}, actual {result}"
        )
