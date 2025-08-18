import random
import unittest
from parameterized import parameterized
import torch
from ..transformer_model import TransformerModel
from ..model_config import ModelConfig, ParallelConfig, QuantConfig
from ..layers.attention import AttentionMetadataTensorCast, AttentionTensorCast
from ..patch_torch import patch_torch

class ModelLoadTestCase(unittest.TestCase):
    @parameterized.expand([
        ["Qwen/Qwen3-32B"],
        # ["Qwen/Qwen3-235B-A22B"],
        # ["deepseek-ai/DeepSeek-V3"],
        ["zai-org/GLM-4.5"],
    ])
    def test_vanilla_transformer_model_eager(self, model_id):
        num_tokens = 100
        model_config = ModelConfig(ParallelConfig(), QuantConfig())
        model = TransformerModel(model_id, model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))

    @parameterized.expand([
        ["Qwen/Qwen3-32B"],
        # ["Qwen/Qwen3-235B-A22B"],
        # ["deepseek-ai/DeepSeek-V3"],
        ["zai-org/GLM-4.5"],
    ])
    def test_vanilla_transformer_model_compile(self, model_id):
        num_tokens = 100
        model_config = ModelConfig(ParallelConfig(), QuantConfig())
        model = TransformerModel(model_id, model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        model_compiled = torch.compile(model, backend="eager")
        with torch.no_grad(), patch_torch():
            outputs = model_compiled.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))

    @parameterized.expand([
        ["Qwen/Qwen3-32B"],
        # ["Qwen/Qwen3-235B-A22B"],
        # ["deepseek-ai/DeepSeek-V3"],
        ["zai-org/GLM-4.5"],
    ])
    def test_prefill_without_kvcache_eager(self, model_id):
        num_tokens = 100
        model_config = ModelConfig(ParallelConfig(), QuantConfig(), attention_cls=AttentionTensorCast)
        model = TransformerModel(model_id, model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad(), patch_torch():
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))

    @parameterized.expand([
        ["Qwen/Qwen3-32B"],
        # ["Qwen/Qwen3-235B-A22B"],
        # ["deepseek-ai/DeepSeek-V3"],
        ["zai-org/GLM-4.5"],
    ])
    def test_prefill_without_kvcache_compile(self, model_id):
        num_tokens = 100
        model_config = ModelConfig(ParallelConfig(), QuantConfig(), attention_cls=AttentionTensorCast)
        model = TransformerModel(model_id, model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        model_compiled = torch.compile(model, backend="eager", fullgraph=True, dynamic=True)
        with torch.no_grad(), patch_torch():
            outputs = model_compiled.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))

    @parameterized.expand([
        ["Qwen/Qwen3-32B", False],
        ["Qwen/Qwen3-32B", True],
        # ["Qwen/Qwen3-235B-A22B"],
        # ["deepseek-ai/DeepSeek-V3"],
        ["zai-org/GLM-4.5", False],
        ["zai-org/GLM-4.5", True],
    ])
    def test_prefill_with_kvcache(self, model_id, do_compile):
        batch_size = 2
        query_len_1 = 55
        query_len_2 = 45
        seq_len_1 = 2000
        seq_len_2 = 1500
        num_blocks = 10000
        block_size = 128
        max_seq_len = max(seq_len_1, seq_len_2)
        query_start_loc = torch.tensor([0, query_len_1, query_len_1 + query_len_2], dtype=torch.long)
        seq_lens = torch.tensor([seq_len_1, seq_len_2], dtype=torch.long)
        max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        block_tables = []
        for _ in range(batch_size):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        block_table_tensor = torch.tensor(block_tables, dtype=torch.long)
        attn_meta = AttentionMetadataTensorCast(query_start_loc=query_start_loc, seq_lens=seq_lens, block_table_tensor=block_table_tensor)

        num_tokens = query_len_1 + query_len_2
        model_config = ModelConfig(ParallelConfig(), QuantConfig(), attention_cls=AttentionTensorCast)
        model = TransformerModel(model_id, model_config)
        if do_compile:
            model = torch.compile(model, backend="eager", dynamic=True, fullgraph=True)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        kv_cache_by_layers = {}
        for i in range(model.num_hidden_layers):
            kv_cache_by_layers[i] = torch.empty(
                [2, num_blocks, block_size, model.text_config.num_key_value_heads, model.text_config.head_dim],
                dtype=model_config.dtype,
                device="meta"
            )

        with torch.no_grad(), patch_torch():
            outputs = model.forward(inputs, position_ids, attention_meta=attn_meta, kv_cache_by_layers=kv_cache_by_layers)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
