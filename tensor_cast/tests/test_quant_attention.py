import random
import unittest

import torch
from parameterized import parameterized

from ..device import A2

from ..layers.attention import AttentionMetadataTensorCast, AttentionTensorCast
from ..model_config import (
    AttentionQuantConfig,
    ModelConfig,
    ParallelConfig,
    QuantConfig,
)
from ..performance_model.analytic import AnalyticPerformanceModel
from ..runtime import Runtime
from ..transformers.model import TransformerModel


def get_quant_config(start_layer_id, end_layer_id):
    quant_config = QuantConfig()
    for i in range(start_layer_id, end_layer_id):
        quant_config.attention_configs[i] = AttentionQuantConfig(
            query_scale=torch.tensor(1.0),
            kv_scale=torch.tensor(1.0),
            attention_prob_scale=torch.tensor(1.0),
        )
    return quant_config


class TestQuantAttention(unittest.TestCase):
    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            # ["Qwen/Qwen3-235B-A22B"],
            # ["deepseek-ai/DeepSeek-V3"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_model_quant_static_int8(self, model_id):
        batch_size = 2
        query_len_1 = 55
        query_len_2 = 45
        seq_len_1 = 2000
        seq_len_2 = 1500
        num_blocks = 10000
        block_size = 128
        max_seq_len = max(seq_len_1, seq_len_2)
        query_start_loc = torch.tensor(
            [0, query_len_1, query_len_1 + query_len_2], dtype=torch.long
        )
        seq_lens = torch.tensor([seq_len_1, seq_len_2], dtype=torch.long)
        max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        block_tables = []
        for _ in range(batch_size):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        block_table_tensor = torch.tensor(block_tables, dtype=torch.long)
        attn_meta = AttentionMetadataTensorCast(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table_tensor=block_table_tensor,
        )

        num_tokens = query_len_1 + query_len_2
        kv_quant_start_idx = 2
        kv_quant_end_idx = 9
        model_config = ModelConfig(
            ParallelConfig(),
            get_quant_config(kv_quant_start_idx, kv_quant_end_idx),
            attention_cls=AttentionTensorCast,
        )
        model = TransformerModel(model_id, model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        kv_cache_by_layers = {}
        for i in range(model.num_hidden_layers):
            kv_cache_by_layers[i] = torch.empty(
                [
                    2,
                    num_blocks,
                    block_size,
                    model.text_config.num_key_value_heads,
                    model.text_config.head_dim,
                ],
                dtype=torch.int8
                if i in range(kv_quant_start_idx, kv_quant_end_idx)
                else model_config.dtype,
                device="meta",
            )

        machine_config = A2
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
            )
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        result = runtime.table_averages()
        self.assertIn("quantize.default", result)
        self.assertIn("reshape_and_cache.default", result)
        self.assertIn("attention_quant.default", result)
