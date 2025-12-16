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

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock, mock_open, call, ANY
from pathlib import Path

# 更新导入路径以匹配您的实际模块位置
from src.optimize.experimental.patch.patch_manager import check_flag, add_patch, Patch2rc1


class TestPatchManager(unittest.TestCase):

    def setUp(self):
        # 创建临时目录
        self.temp_dir = Path(tempfile.mkdtemp())

        # 创建模拟的 mindie_llm 模块
        self.mock_mindie_llm = MagicMock()
        sys.modules['mindie_llm'] = self.mock_mindie_llm

    def tearDown(self):
        # 安全地删除整个临时目录
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # 清理模拟模块
        if 'mindie_llm' in sys.modules:
            del sys.modules['mindie_llm']

    # 重构文件检查测试 - 使用真实文件
    def test_check_flag_no_match(self):
        """测试完全不匹配的情况"""
        # 创建真实文件
        target_file = self.temp_dir / "target.txt"
        patch_file = self.temp_dir / "patch.txt"

        with open(target_file, "w", encoding="utf-8") as f:
            f.write("line1\nline2\n")

        with open(patch_file, "w", encoding="utf-8") as f:
            f.write("patch1\npatch2\n")

        result = check_flag(str(target_file), str(patch_file))
        self.assertTrue(result, "内容完全不匹配时应返回True")

    def test_check_flag_partial_match(self):
        """测试部分匹配的情况"""
        # 创建真实文件
        target_file = self.temp_dir / "target.txt"
        patch_file = self.temp_dir / "patch.txt"

        # 目标文件包含部分补丁内容
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("start\n")
            f.write("patch_line1: some code 123\n")  # 补丁的第一行
            f.write("middle\n")
            # 缺少第二行补丁内容
            f.write("end\n")

        with open(patch_file, "w", encoding="utf-8") as f:
            f.write("patch_line1: some code 123\n")
            f.write("patch_line2: more code 456\n")

        result = check_flag(str(target_file), str(patch_file))
        self.assertTrue(result, "部分匹配时应返回True")

    def test_check_flag_full_match(self):
        """测试完全匹配的情况"""
        # 创建真实文件
        target_file = self.temp_dir / "target.txt"
        patch_file = self.temp_dir / "patch.txt"

        with open(target_file, "w", encoding="utf-8") as f:
            f.write("patch1\npatch2\nfooter\n")

        with open(patch_file, "w", encoding="utf-8") as f:
            f.write("patch1\npatch2\n")

        result = check_flag(str(target_file), str(patch_file))
        self.assertFalse(result, "补丁内容在目标文件中完全匹配时，预期返回False(表示已存在)但实际返回了True")

    # 测试版本检查
    def test_check_version_low_warning(self):
        """测试版本过低的情况"""
        with patch('src.optimize.experimental.patch.patch_manager.logger.warning') as mock_warning:
            Patch2rc1.check_version("1.0")
            mock_warning.assert_called_once_with("The version may not match.")

    def test_check_version_high_warning(self):
        """测试版本过高的情况"""
        with patch('src.optimize.experimental.patch.patch_manager.logger.warning') as mock_warning:
            Patch2rc1.check_version("3.0")
            mock_warning.assert_called_once_with("The version may not match.")

    def test_check_version_exactly_low(self):
        """测试精确匹配下限版本的情况"""
        with patch('src.optimize.experimental.patch.patch_manager.logger.warning') as mock_warning:
            Patch2rc1.check_version(Patch2rc1.mindie_llm_low)
            mock_warning.assert_called_once_with("The version may not match.")

    def test_check_version_exactly_up(self):
        """测试精确匹配上限版本的情况"""
        with patch('src.optimize.experimental.patch.patch_manager.logger.warning') as mock_warning:
            Patch2rc1.check_version(Patch2rc1.mindie_llm)
            mock_warning.assert_not_called()

    def test_check_version_patch_level(self):
        """测试补丁版本比较"""
        with patch('src.optimize.experimental.patch.patch_manager.logger.warning') as mock_warning:
            # 注意：2.0.1 > 2.0，所以应该在范围外
            Patch2rc1.check_version("2.0.1")
            mock_warning.assert_called_once_with("The version may not match.")
