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

from src.optimize.experimental.patch import enable_patch, env_patch, vllm_env_patch


# 导入被测模块
sys.modules['modelevalstate'] = MagicMock()


class TestEnablePatch(unittest.TestCase):
    def setUp(self):
        """在每个测试前重置全局状态"""
        self.reset_globals()
        self.addCleanup(self.reset_globals)

        # 模拟 logger
        self.logger_patcher = patch(
            'src.optimize.experimental.patch.logger',
            create=True, new_callable=MagicMock
        )
        self.mock_logger = self.logger_patcher.start()
        self.addCleanup(self.logger_patcher.stop)

        # 正确模拟 warnings.warn
        self.warn_patcher = patch(
            'warnings.warn',  # 使用正确的路径
            create=True, new_callable=MagicMock
        )
        self.mock_warn = self.warn_patcher.start()
        self.addCleanup(self.warn_patcher.stop)

    def reset_globals(self):
        # 重置 patch 列表
        env_patch.clear()
        vllm_env_patch.clear()

        # 重建初始 patch 字典结构
        env_patch.update({
            "MODEL_EVAL_STATE_COLLECT": [],
            "MODEL_EVAL_STATE_SIMULATE": [],
            "MODEL_EVAL_STATE_ALL": [],
            "MODEL_EVAL_STATE_COLLECT_ELEGANT": [],
            "MODEL_EVAL_STATE_SIMULATE_ELEGANT": [],
            "MODEL_EVAL_STATE_ALL_ELEGANT": []
        })
        vllm_env_patch.update({
            "MODEL_EVAL_STATE_COLLECT": [],
            "MODEL_EVAL_STATE_SIMULATE": [],
            "MODEL_EVAL_STATE_ALL": [],
            "MODEL_EVAL_STATE_COLLECT_ELEGANT": [],
            "MODEL_EVAL_STATE_SIMULATE_ELEGANT": [],
            "MODEL_EVAL_STATE_ALL_ELEGANT": [],
        })

    def test_no_patches_available(self):
        """测试没有可用补丁的情况"""
        enable_patch("MODEL_EVAL_STATE_SIMULATE")
        self.mock_logger.info.assert_not_called()

    @patch('src.optimize.experimental.patch.get_module_version')
    def test_mindie_successful_patch(self, mock_get_version):
        """测试mindie成功应用补丁"""
        mock_patch = MagicMock()
        mock_patch.check_version.return_value = True
        env_patch["MODEL_EVAL_STATE_SIMULATE"].append(mock_patch)

        mock_get_version.return_value = "1.0.0"
        enable_patch("MODEL_EVAL_STATE_SIMULATE")

        mock_patch.check_version.assert_called_once()
        mock_patch.patch.assert_called_once()
        self.mock_logger.info.assert_called_once()

    @patch('src.optimize.experimental.patch.get_module_version')
    def test_vllm_successful_patch(self, mock_get_version):
        """测试vllm_ascend成功应用补丁"""
        mock_patch = MagicMock()
        mock_patch.check_version.return_value = True
        vllm_env_patch["MODEL_EVAL_STATE_SIMULATE"].append(mock_patch)

        mock_get_version.return_value = "2.3.4"
        enable_patch("MODEL_EVAL_STATE_SIMULATE")

        mock_patch.check_version.assert_called_once_with("2.3.4")
        mock_patch.patch.assert_called_once()
        self.mock_logger.info.assert_called_once()

    @patch('src.optimize.experimental.patch.get_module_version')
    def test_combination_patch_success(self, mock_get_version):
        """测试mindie和vllm同时成功应用补丁"""
        # 准备 mindie 的补丁
        mindie_patch = MagicMock()
        mindie_patch.check_version.return_value = True
        env_patch["MODEL_EVAL_STATE_SIMULATE"].append(mindie_patch)

        # 准备 vllm 的补丁
        vllm_patch = MagicMock()
        vllm_patch.check_version.return_value = True
        vllm_env_patch["MODEL_EVAL_STATE_SIMULATE"].append(vllm_patch)

        mock_get_version.side_effect = ["1.2.3", "4.5.6"]
        enable_patch("MODEL_EVAL_STATE_SIMULATE")

        mindie_patch.patch.assert_called_once()
        vllm_patch.patch.assert_called_once()
        self.mock_logger.info.assert_called_once()
        self.assertIn("patch list", self.mock_logger.info.call_args[0][0])

    @patch('src.optimize.experimental.patch.get_module_version')
    def test_version_check_failure(self, mock_get_version):
        """测试版本检查失败"""
        mock_patch = MagicMock()
        mock_patch.check_version.return_value = False
        env_patch["MODEL_EVAL_STATE_SIMULATE"].append(mock_patch)

        mock_get_version.return_value = "1.0.0"
        enable_patch("MODEL_EVAL_STATE_SIMULATE")

        mock_patch.patch.assert_not_called()
        self.mock_logger.info.assert_not_called()

    @patch('src.optimize.experimental.patch.get_module_version')
    def test_mixed_success_failure(self, mock_get_version):
        """测试部分补丁成功、部分失败的情况"""
        # 准备 mindie 的补丁 - 成功
        good_patch = MagicMock()
        good_patch.check_version.return_value = True

        # 准备 mindie 的补丁 - 失败
        bad_patch = MagicMock()
        bad_patch.check_version.return_value = False

        env_patch["MODEL_EVAL_STATE_SIMULATE"].extend([good_patch, bad_patch])

        # 准备 vllm 的补丁 - 版本检查失败
        vllm_patch = MagicMock()
        vllm_patch.check_version.return_value = False
        vllm_env_patch["MODEL_EVAL_STATE_SIMULATE"].append(vllm_patch)

        mock_get_version.side_effect = ["1.0.0", "2.0.0"]
        enable_patch("MODEL_EVAL_STATE_SIMULATE")

        good_patch.patch.assert_called_once()
        bad_patch.patch.assert_not_called()
        vllm_patch.patch.assert_not_called()
        self.mock_logger.info.assert_called_once()

    def test_target_env_not_found(self):
        """测试目标环境未找到的情况"""
        mock_patch = MagicMock()
        mock_patch.check_version.return_value = True
        env_patch["MODEL_EVAL_STATE_SIMULATE"].append(mock_patch)

        # 使用不存在的目标环境
        enable_patch("INVALID_ENV")

        mock_patch.patch.assert_not_called()
        self.mock_logger.info.assert_not_called()

    @patch('src.optimize.experimental.patch.get_module_version')
    def test_all_target_envs(self, mock_get_version):
        """测试所有可能的目标环境"""
        target_envs = [
            "MODEL_EVAL_STATE_COLLECT",
            "MODEL_EVAL_STATE_SIMULATE",
            "MODEL_EVAL_STATE_ALL",
            "MODEL_EVAL_STATE_COLLECT_ELEGANT",
            "MODEL_EVAL_STATE_SIMULATE_ELEGANT",
            "MODEL_EVAL_STATE_ALL_ELEGANT"
        ]

        mock_get_version.return_value = "1.0.0"

        for env in target_envs:
            with self.subTest(env=env):
                # 重置 vllm_env_patch 确保包含所有键
                vllm_env_patch[env] = []

                # 为当前环境添加 mock_patch
                mock_patch = MagicMock()
                mock_patch.check_version.return_value = True
                env_patch[env] = [mock_patch]

                enable_patch(env)

                mock_patch.patch.assert_called_once()
                mock_patch.reset_mock()

    @patch('src.optimize.experimental.patch.get_module_version')
    def test_multiple_vllm_patches(self, mock_get_version):
        """测试多个vllm补丁同时应用"""
        # 准备多个 vllm 补丁
        patch1 = MagicMock()
        patch1.check_version.return_value = True

        patch2 = MagicMock()
        patch2.check_version.return_value = True

        vllm_env_patch["MODEL_EVAL_STATE_ALL"].extend([patch1, patch2])

        mock_get_version.return_value = "3.0.0"
        enable_patch("MODEL_EVAL_STATE_ALL")

        patch1.patch.assert_called_once()
        patch2.patch.assert_called_once()
        self.mock_logger.info.assert_called_once()

    @patch('src.optimize.experimental.patch.get_module_version')
    def test_value_error_handling(self, mock_get_version):
        """测试处理get_module_version的ValueError"""
        mock_patch = MagicMock()
        env_patch["MODEL_EVAL_STATE_SIMULATE"].append(mock_patch)

        mock_get_version.side_effect = ValueError("Invalid version")
        enable_patch("MODEL_EVAL_STATE_SIMULATE")

        # 不应记录任何错误（被安全捕获）
        self.mock_warn.assert_not_called()
        self.mock_logger.info.assert_not_called()

    def test_no_logging_when_no_patch_applied(self):
        """测试当没有补丁应用时不会记录日志"""
        mock_patch = MagicMock()
        mock_patch.check_version.return_value = False
        env_patch["MODEL_EVAL_STATE_SIMULATE"].append(mock_patch)

        enable_patch("MODEL_EVAL_STATE_SIMULATE")

        self.mock_logger.info.assert_not_called()

    @patch('src.optimize.experimental.patch.get_module_version')
    def test_patch_class_repr_in_log(self, mock_get_version):
        """测试日志中包含补丁类的表示"""
        # 确保有日志调用
        mock_patch = MagicMock()
        mock_patch.__repr__ = MagicMock(return_value="<TestPatchClass>")
        mock_patch.check_version.return_value = True
        mock_patch.patch.return_value = None

        # 模拟版本获取
        mock_get_version.return_value = "1.0.0"

        # 设置环境
        env_patch["MODEL_EVAL_STATE_SIMULATE"] = [mock_patch]
        vllm_env_patch["MODEL_EVAL_STATE_SIMULATE"] = []

        # 重置日志模拟
        self.mock_logger.reset_mock()

        enable_patch("MODEL_EVAL_STATE_SIMULATE")

        # 确保日志被调用
        self.assertTrue(self.mock_logger.info.called)

        # 查找参数中的表示
        call_args = self.mock_logger.info.call_args[0][0]
        self.assertIn("<TestPatchClass>", call_args)

    @patch('src.optimize.experimental.patch.get_module_version')
    def test_elegant_env_patch(self, mock_get_version):
        """测试优雅环境的补丁应用"""
        mock_patch = MagicMock()
        mock_patch.check_version.return_value = True

        # 确保同时设置 env_patch 和 vllm_env_patch
        env = "MODEL_EVAL_STATE_ALL_ELEGANT"
        env_patch[env] = [mock_patch]
        vllm_env_patch[env] = []

        mock_get_version.return_value = "1.0.0"
        enable_patch(env)

        mock_patch.patch.assert_called_once()
