# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock

from serving_cast.service.backend_factory import StrategyFactory
from serving_cast.service.mindie_backend import MindIEAggBackend


class TestStrategyFactory(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_args = Mock()
        self.mock_args.disaggregation = False

        self.mock_model = Mock()
        self.mock_model.model_config = Mock()

    def test_create_backend_mindie_aggregation(self):
        """Test creating MindIE aggregation backend"""
        backend = StrategyFactory.create_backend(
            "mindie", self.mock_args, self.mock_model
        )

        self.assertIsInstance(backend, MindIEAggBackend)
        self.assertEqual(backend.name, "mindie_aggregation")

    def test_create_backend_case_insensitive(self):
        """Test that backend creation is case insensitive"""
        backend_upper = StrategyFactory.create_backend(
            "MINDIE", self.mock_args, self.mock_model
        )
        self.assertIsInstance(backend_upper, MindIEAggBackend)

        backend_mixed = StrategyFactory.create_backend(
            "MiNdIe", self.mock_args, self.mock_model
        )
        self.assertIsInstance(backend_mixed, MindIEAggBackend)

    def test_create_backend_without_model(self):
        """Test creating backend without providing a model"""
        backend = StrategyFactory.create_backend("mindie", self.mock_args, None)

        self.assertIsInstance(backend, MindIEAggBackend)
        self.assertEqual(backend.name, "mindie_aggregation")

    def test_create_backend_unsupported_backend(self):
        """Test creating an unsupported backend raises ValueError"""
        with self.assertRaises(ValueError) as context:
            StrategyFactory.create_backend(
                "unsupported_backend", self.mock_args, self.mock_model
            )

        self.assertIn("Unsupported backend", str(context.exception))
        self.assertIn("unsupported_backend", str(context.exception))

    def test_frameworks_cls_contains_mindie_backend(self):
        """Test that the frameworks class dictionary contains MindIE backend"""
        from serving_cast.service.mindie_backend import MindIEAggBackend

        expected_key = MindIEAggBackend.name
        self.assertIn(expected_key, StrategyFactory._frameworks_cls)
        self.assertEqual(
            StrategyFactory._frameworks_cls[expected_key], MindIEAggBackend
        )

    def test_initialize_method_called(self):
        """Test that initialize method is called when model is provided"""
        with unittest.mock.patch.object(MindIEAggBackend, "initialize") as mock_init:
            _ = StrategyFactory.create_backend(
                "mindie", self.mock_args, self.mock_model
            )

            mock_init.assert_called_once_with(self.mock_args, self.mock_model)


if __name__ == "__main__":
    unittest.main()
