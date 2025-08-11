import unittest
import torch
from ..transformer_model import TransformerModel, ModelConfig, ParallelConfig, QuantConfig
from ..attention import AttentionMetadataTensorCast

class ModelLoadTestCase(unittest.TestCase):
    def test_prefill_without_kvcache_eager(self):
        num_tokens = 100
        model_id = "Qwen/Qwen3-32B"
        model_config = ModelConfig(ParallelConfig(), QuantConfig())
        model = TransformerModel(model_id, model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with torch.no_grad():
            outputs = model.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))

    def test_prefill_without_kvcache_compile(self):
        num_tokens = 100
        model_id = "Qwen/Qwen3-32B"
        model_config = ModelConfig(ParallelConfig(), QuantConfig())
        model = TransformerModel(model_id, model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        model_compiled = torch.compile(model, backend="eager")
        with torch.no_grad():
            outputs = model_compiled.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))

    def test_prefill_with_kvcache(self):
        batch_size = 2
        query_len_1 = 55
        query_len_2 = 45
        seq_len_1 = 2000
        seq_len_2 = 1500
        query_start_loc = torch.tensor([0, query_len_1, query_len_1 + query_len_2], dtype=torch.long)
        seq_lens = torch.tensor([seq_len_1, seq_len_2], dtype=torch.long)
        attn_meta = AttentionMetadataTensorCast(query_start_loc=query_start_loc, seq_lens=seq_lens)

        num_tokens = query_len_1 + query_len_2
        model_id = "Qwen/Qwen3-32B"
        model_config = ModelConfig(ParallelConfig(), QuantConfig())
        model = TransformerModel(model_id, model_config)
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        
        with torch.no_grad():
            outputs = model.forward(inputs, position_ids, attention_meta=attn_meta)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))