import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from tensor_cast.layers.mla import DeepseekSparseAttentionIndexer


class TestDeepseekSparseAttentionIndexer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10

        inner_module = nn.Module()
        inner_module.hidden_size = 16
        inner_module.num_heads = 4
        inner_module.head_dim = 8
        inner_module.qk_rope_head_dim = 4
        inner_module.index_topk = 2
        inner_module.q_lora_rank = 4

        inner_module.wq_b = nn.Linear(
            inner_module.q_lora_rank,
            inner_module.num_heads * inner_module.head_dim,
            bias=False,
        )
        inner_module.wk = nn.Linear(
            inner_module.hidden_size, inner_module.head_dim, bias=False
        )
        inner_module.k_norm = nn.LayerNorm(inner_module.head_dim)
        inner_module.weights_proj = nn.Linear(
            inner_module.hidden_size, inner_module.num_heads, bias=False
        )
        inner_module.softmax_scale = inner_module.head_dim**-0.5

        self.indexer = DeepseekSparseAttentionIndexer(inner_module)

        self.hidden_states = torch.randn(
            self.batch_size, self.seq_len, inner_module.hidden_size
        )
        self.qa_normed = torch.randn(
            self.batch_size, self.seq_len, inner_module.q_lora_rank
        )
        self.position_embeddings = (
            torch.randn(self.seq_len, inner_module.qk_rope_head_dim),
            torch.randn(self.seq_len, inner_module.qk_rope_head_dim),
        )
        self.indexer_cache = None

    @patch("torch.ops.tensor_cast.dsa_index")
    @patch("torch.ops.tensor_cast.dsa_index_cache")
    def test_forward(self, mock_dsa_index_cache, mock_dsa_index):
        mock_dsa_index.return_value = torch.randn(
            self.batch_size, self.seq_len, self.indexer.index_topk
        )
        mock_dsa_index_cache.return_value = None

        res = self.indexer.forward(
            self.hidden_states,
            self.qa_normed,
            self.position_embeddings,
            self.indexer_cache,
        )

        self.assertEqual(
            res.shape, (self.batch_size, self.seq_len, self.indexer.index_topk)
        )
        mock_dsa_index.assert_called_once()
        mock_dsa_index_cache.assert_called_once()
