# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import argparse
import logging
import unittest

from serving_cast.service.utils import (
    BackendName,
    check_positive_float,
    check_positive_integer,
    check_string_valid,
    DataConfig,
    set_logger,
)


class TestServiceUtils(unittest.TestCase):
    def test_check_string_valid_within_limit_and_valid_chars(self):
        """Test check_string_valid with valid string"""
        valid_string = "valid_string123/test-path.file"
        result = check_string_valid(valid_string, max_len=100)
        self.assertEqual(result, valid_string)

    def test_check_positive_integer_valid(self):
        """Test check_positive_integer with valid integers"""
        self.assertEqual(check_positive_integer("1"), 1)
        self.assertEqual(check_positive_integer("100"), 100)
        self.assertEqual(check_positive_integer(5), 5)

    def test_check_positive_integer_invalid_string(self):
        """Test check_positive_integer with invalid string"""
        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_integer("abc")

    def test_check_positive_integer_non_positive(self):
        """Test check_positive_integer with non-positive values"""
        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_integer("0")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_integer("-1")

    def test_check_positive_integer_too_large(self):
        """Test check_positive_integer with very large value"""
        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_integer("2000000")  # Greater than 1e6

    def test_check_positive_float_valid(self):
        """Test check_positive_float with valid floats"""
        self.assertEqual(check_positive_float("1.5"), 1.5)
        self.assertEqual(check_positive_float("100"), 100.0)
        self.assertEqual(check_positive_float("inf"), float("inf"))
        self.assertEqual(check_positive_float("INF"), float("inf"))

    def test_check_positive_float_invalid(self):
        """Test check_positive_float with invalid values"""
        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_float("abc")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_float("0")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_float("-1.5")

    def test_data_config_creation(self):
        """Test DataConfig creation with default values"""
        config = DataConfig()
        self.assertIsNone(config.input_length)
        self.assertIsNone(config.output_length)
        self.assertIsNone(config.device_profile)

    def test_backend_name_enum(self):
        """Test BackendName enum values"""
        self.assertEqual(BackendName.MindIE.value, "MindIE")

    def test_set_logger_functionality(self):
        """Test set_logger function sets up logger properly"""
        test_logger = logging.getLogger("test_logger")

        # Clear any existing handlers
        test_logger.handlers.clear()

        set_logger(test_logger)

        # Verify logger properties
        self.assertFalse(test_logger.propagate)
        self.assertEqual(test_logger.level, logging.INFO)
        self.assertTrue(len(test_logger.handlers) > 0)

        # Clean up
        test_logger.handlers.clear()
