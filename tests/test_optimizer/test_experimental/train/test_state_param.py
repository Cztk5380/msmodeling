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
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from dataclasses import asdict
import shutil

from src.optimize.experimental.train.state_param import StateParam


class TestStateParam(unittest.TestCase):
    """测试 StateParam 配置类的功能"""

    def setUp(self):
        # 创建临时测试目录
        self.test_dir = Path("test_state_param")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        # 创建配置实例
        param = StateParam(
            title="Test Config",
            base_path=Path("/tmp/test"),
            xgb_model_train_param={'learning_rate': 0.1}
        )

        # 验证自动创建的路径
        self.assertEqual(param.model_dir, Path("/tmp/test/model"))
        self.assertEqual(param.step_dir, Path("/tmp/test/step"))
        self.assertEqual(param.bak_dir, Path("/tmp/test/bak"))

        # 验证路径确实被创建
        self.assertTrue(param.model_dir.exists())
        self.assertTrue(param.step_dir.exists())

        # 清理测试目录
        param.model_dir.rmdir()
        param.step_dir.rmdir()
        param.bak_dir.rmdir()
        param.base_path.rmdir()

    def test_asdict(self):
        # 创建配置实例
        param = StateParam(
            title="Test Config",
            base_path=Path("/tmp/test")
        )

        # 转换为字典
        param_dict = asdict(param)

        # 验证转换结果
        self.assertEqual(param_dict['title'], "Test Config")
        self.assertEqual(param_dict['predict_field'], "model_execute_time")
        self.assertIsInstance(param_dict['model_dir'], Path)

    def test_default_initialization(self):
        """测试默认参数初始化"""
        sp = StateParam(base_path=self.test_dir)

        # 验证基本属性
        self.assertEqual(sp.title, "MixModel")
        self.assertEqual(sp.base_path, self.test_dir)
        self.assertEqual(sp.predict_field, "model_execute_time")
        self.assertTrue(sp.save_model)
        self.assertFalse(sp.shuffle)

        # 验证路径创建
        self.assertTrue(sp.model_dir.exists())
        self.assertTrue(sp.step_dir.exists())
        self.assertTrue(sp.bak_dir.exists())

        # 验证XGBoost模型路径
        self.assertEqual(sp.xgb_model_save_model_path, sp.model_dir / "xgb_model.ubj")
        self.assertEqual(sp.xgb_model_load_model_path, sp.model_dir / "xgb_model.ubj")

    def test_custom_initialization(self):
        """测试自定义参数初始化"""
        custom_sp = StateParam(
            base_path=self.test_dir / "custom",
            title="CustomModel",
            predict_field="custom_field",
            save_model=False,
            shuffle=False,
            plot_pred_and_real=False,
            op_algorithm="scale"
        )

        # 验证自定义属性
        self.assertEqual(custom_sp.title, "CustomModel")
        self.assertEqual(custom_sp.predict_field, "custom_field")
        self.assertFalse(custom_sp.save_model)
        self.assertFalse(custom_sp.shuffle)
        self.assertFalse(custom_sp.plot_pred_and_real)
        self.assertEqual(custom_sp.op_algorithm, "scale")

        # 验证自定义路径
        self.assertEqual(custom_sp.base_path, self.test_dir / "custom")
        self.assertTrue((self.test_dir / "custom").exists())

    @patch("src.optimize.experimental.train.state_param.Path.mkdir")
    def test_directory_creation_failure(self, mock_mkdir):
        """测试目录创建失败处理"""
        mock_mkdir.side_effect = OSError("Permission denied")

        with self.assertRaises(OSError):
            sp = StateParam(base_path=self.test_dir / "invalid")

    def test_xgb_params(self):
        """测试XGBoost参数配置"""
        sp = StateParam(
            base_path=self.test_dir,
            xgb_model_train_param={"max_depth": 6, "eta": 0.3},
            xgb_model_update_param={"updater": "refresh"}
        )

        # 验证训练参数
        self.assertEqual(sp.xgb_model_train_param["max_depth"], 6)
        self.assertEqual(sp.xgb_model_train_param["eta"], 0.3)

        # 验证更新参数
        self.assertEqual(sp.xgb_model_update_param["updater"], "refresh")


if __name__ == '__main__':
    unittest.main()
