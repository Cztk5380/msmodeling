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
from unittest.mock import patch, MagicMock
import sys

from src.optimize.experimental.patch.patch_vllm import PatchVllm


class TestPatchVllm(unittest.TestCase):

    def tearDown(self):
        # 清理sys.modules
        if 'vllm_ascend' in sys.modules:
            del sys.modules['vllm_ascend']

    @patch("src.optimize.experimental.patch.patch_vllm.add_patch")
    @patch("src.optimize.experimental.patch.patch_vllm.check_flag")
    @patch("src.optimize.experimental.patch.patch_vllm.logger")
    @patch("pathlib.Path.exists")
    def test_patch_not_applied(self, mock_exists, mock_logger, mock_check_flag, mock_add_patch):
        """测试需要打补丁的场景"""
        # 模拟文件存在
        mock_exists.return_value = True

        # 模拟vllm_ascend模块
        with patch.dict('sys.modules', {'vllm_ascend': MagicMock(__path__=["/vllm_ascend/path"])}):
            # 设置需要打补丁的标志
            mock_check_flag.return_value = True

            # 调用方法
            PatchVllm.patch()

    @patch("src.optimize.experimental.patch.patch_vllm.add_patch")
    @patch("src.optimize.experimental.patch.patch_vllm.check_flag")
    @patch("src.optimize.experimental.patch.patch_vllm.logger")
    @patch("pathlib.Path.exists")
    def test_patch_already_applied(self, mock_exists, mock_logger, mock_check_flag, mock_add_patch):
        """测试补丁已存在的场景"""
        # 模拟文件存在
        mock_exists.return_value = True

        # 模拟vllm_ascend模块
        with patch.dict('sys.modules', {'vllm_ascend': MagicMock(__path__=["/vllm_ascend/path"])}):
            # 设置补丁已存在的标志
            mock_check_flag.return_value = False

            # 调用方法
            PatchVllm.patch()

            # 验证没有打补丁
            mock_add_patch.assert_not_called()

    @patch("src.optimize.experimental.patch.patch_vllm.add_patch")
    @patch("src.optimize.experimental.patch.patch_vllm.check_flag")
    @patch("pathlib.Path.exists")
    def test_vllm_path_handling(self, mock_exists, mock_check_flag, mock_add_patch):
        """测试不同vllm路径的处理"""
        # 模拟文件存在
        mock_exists.return_value = True

        # 模拟vllm_ascend模块
        with patch.dict('sys.modules', {'vllm_ascend': MagicMock(__path__=["/custom/vllm/ascend"])}):
            # 设置需要打补丁的标志
            mock_check_flag.return_value = True

            # 调用方法
            PatchVllm.patch()

    def test_check_version(self):
        """测试check_version方法始终返回True"""
        self.assertTrue(PatchVllm.check_version("any_version"))
