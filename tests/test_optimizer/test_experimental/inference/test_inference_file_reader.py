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
import unittest
from pathlib import Path
from unittest.mock import patch

from src.optimize.experimental.inference.file_reader import FileHanlder, StaticFile


class TestFileHandler:
    @staticmethod
    def test_load_static_data(static_file):
        fh = FileHanlder(static_file)
        fh.load_static_data()
        assert fh.hardware
        assert fh.env_info
        assert fh.mindie_info
        assert fh.model_config_info
        assert fh.model_struct_info
        assert fh.prefill_op_data
        assert fh.decode_op_data


class TestStaticFile(unittest.TestCase):
    def setUp(self):
        self.base_path = Path("data/model")

    def test_post_init_base_path_not_exists(self):
        with patch('pathlib.Path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                StaticFile(base_path=self.base_path)

    def test_post_init_base_path_exists(self):
        with patch('pathlib.Path.exists', return_value=True):
            static_file = StaticFile(base_path=self.base_path)
            self.assertEqual(static_file.hardware_path, self.base_path.joinpath("hardware.json"))
            self.assertEqual(static_file.env_path, self.base_path.joinpath("env.json"))
            self.assertEqual(static_file.mindie_config_path, self.base_path.joinpath("mindie_config.json"))
            self.assertEqual(static_file.config_path, self.base_path.joinpath("model_config.json"))
            self.assertEqual(static_file.model_struct_path, self.base_path.joinpath("model_struct.csv"))
            self.assertEqual(static_file.model_decode_op_path, self.base_path.joinpath("model_decode_op.csv"))
            self.assertEqual(static_file.model_prefill_op_path, self.base_path.joinpath("model_prefill_op.csv"))

    def test_post_init_all_paths_exist(self):
        with patch('pathlib.Path.exists', return_value=True):
            static_file = StaticFile(base_path=self.base_path)
            for path in [static_file.hardware_path, static_file.env_path, static_file.mindie_config_path, \
                         static_file.config_path, static_file.model_struct_path, static_file.model_decode_op_path, \
                         static_file.model_prefill_op_path]:
                self.assertTrue(path.exists())
