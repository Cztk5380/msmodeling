import unittest

import torch
from parameterized import parameterized

from ..compilation import get_backend
from ..compilation.pattern_manager import PatternManager

from ..model_config import ModelConfig, ParallelConfig, QuantConfig
from ..patch_torch import patch_torch
from ..transformer_model import TransformerModel


class PatternReplaceTestCase(unittest.TestCase):
    def setUp(self):
        num_tokens = 100
        self.compile_backend = get_backend()
        with torch.device("meta"):
            self.inputs = torch.empty([1, num_tokens], dtype=torch.long)
            self.position_ids = torch.empty([1, num_tokens], dtype=torch.long)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            # ["Qwen/Qwen3-235B-A22B"],
            # ["deepseek-ai/DeepSeek-V3"],
            # ["zai-org/GLM-4.5"],
        ]
    )
    def test_rms_norm_pattern(self, model_id):
        num_tokens = 100
        model_config = ModelConfig(ParallelConfig(), QuantConfig())
        model = TransformerModel(model_id, model_config)
        self.assertTrue(
            PatternManager.has_pattern(f"rms_norm_pattern_{model_config.dtype}")
        )
        model = torch.compile(
            model, backend=self.compile_backend, fullgraph=True, dynamic=True
        )
        # Comiple the FX graph by first run and Check output shape
        with torch.no_grad(), patch_torch():
            outputs = model.forward(self.inputs, self.position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, model.hidden_size))
        # Check if the pattern is replaced in the FX graph
        graph = self.compile_backend.get_graph()
        graph_str = str(graph.graph)
        self.assertIn("rms_norm", graph_str)
