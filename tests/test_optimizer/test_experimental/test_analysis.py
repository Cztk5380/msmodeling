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
from pathlib import Path

import pandas as pd
import numpy as np

# 导入需要的类和函数
from src.optimize.experimental.analysis import (
    AnalysisState,
    PlotConfig,
    State
)


# 测试类
class TestAnalysisState(unittest.TestCase):
    def setUp(self):
        # 创建测试用的State数据
        self.test_data = {
            State(batch_prefill=1): [10.0, 10.5, 11.0],
            State(batch_prefill=2): [20.0, 20.5, 21.0],
            State(batch_decode=3): [30.0, 30.5, 31.0],
            State(batch_decode=4): [40.0, 40.5, 41.0],
        }

        self.single_data = {
            State(batch_prefill=1): [10.0],
            State(batch_prefill=2): [20.0],
        }

        # 保存路径
        self.save_path = Path("/tmp/test_save_path")
        self.save_path.mkdir(exist_ok=True, parents=True)

    def tearDown(self):
        # 清理测试文件
        for file in self.save_path.iterdir():
            if file.is_file():
                file.unlink()
        self.save_path.rmdir()

    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_computer_mean_sigma(self, mock_close, mock_show, mock_plot):
        # 使用不同的键值组合创建测试数据
        test_data = {
            State(batch_prefill=1): [10.0, 10.5, 11.0],
            State(batch_prefill=1, batch_decode=10): [10.1, 10.6, 11.1],
            State(batch_prefill=2): [20.0, 20.5, 21.0],
            State(batch_prefill=2, batch_decode=20): [20.1, 20.6, 21.1],
        }

        # 手动计算预期的分组数据
        group1_data = [10.0, 10.5, 11.0, 10.1, 10.6, 11.1]
        group2_data = [20.0, 20.5, 21.0, 20.1, 20.6, 21.1]

        # 计算预期的平均值
        expected_mean1 = np.mean(group1_data)
        expected_mean2 = np.mean(group2_data)

        # 调用计算方法
        x, mean, pos_sigma, neg_sigma = AnalysisState.computer_mean_sigma(
            test_data, "batch_prefill"
        )

        # 验证返回值的类型和结构
        self.assertIsInstance(x, list)
        self.assertIsInstance(mean, list)
        self.assertIsInstance(pos_sigma, list)
        self.assertIsInstance(neg_sigma, list)

        # 验证分组数量
        self.assertEqual(len(x), 2)

        # 验证计算结果
        self.assertAlmostEqual(mean[0], expected_mean1, places=2)
        self.assertAlmostEqual(mean[1], expected_mean2, places=2)

    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_input_velocity(self, mock_close, mock_savefig, mock_ylabel,
                                 mock_xlabel, mock_title, mock_grid, mock_legend,
                                 mock_plot):
        # 配置绘图参数
        config = PlotConfig(
            data=self.test_data,
            x_field="batch_prefill",
            title="Test Plot",
            x_label="Batch Size",
            y_label="Latency (ms)",
            save_path=str(self.save_path)
        )

        # 调用绘图方法
        AnalysisState.plot_input_velocity(config)

        # 验证是否调用了绘图函数
        self.assertEqual(mock_plot.call_count, 3)  # 三条线：均值、上界、下界
        mock_title.assert_called_once_with("Test Plot")
        mock_xlabel.assert_called_once_with("Batch Size")
        mock_ylabel.assert_called_once_with("Latency (ms)")
        mock_legend.assert_called_once()
        mock_grid.assert_called_once()

        # 验证保存文件
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

        # 测试无保存路径的情况 (显示图像)
        config.save_path = None
        mock_plot.reset_mock()
        mock_show = MagicMock()
        with patch('matplotlib.pyplot.show', mock_show):
            AnalysisState.plot_input_velocity(config)
            mock_show.assert_called_once()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.show')
    def test_plot_pred_and_real(self, mock_show, mock_close, mock_savefig,
                                mock_legend, mock_ylabel, mock_xlabel,
                                mock_title, mock_scatter, mock_figure):
        # 创建测试数据
        pred = [1.1, 2.1, 3.1]
        real = [1.0, 2.0, 3.0]

        # 测试保存图片的情况
        AnalysisState.plot_pred_and_real(pred, real, self.save_path)

        # 验证绘图函数调用
        self.assertEqual(mock_scatter.call_count, 2)  # pred和real
        mock_title.assert_called_once_with("predict value and real value")
        mock_xlabel.assert_called_once_with("index")
        mock_ylabel.assert_called_once_with("value")
        mock_legend.assert_called_once()
        mock_savefig.assert_called_once_with(self.save_path / "predict value and real value.png")
        mock_close.assert_called_once()

        # 测试无保存路径的情况 (显示图像)
        mock_scatter.reset_mock()
        mock_savefig.reset_mock()
        AnalysisState.plot_pred_and_real(pred, real, None)
        mock_show.assert_called_once()

    def test_std_calculations(self):
        # 创建需要测试的数据
        test_data = {
            State(batch_prefill=1): [1.0, 2.0, 3.0]
        }

        # 调用计算方法
        x, mean, pos_sigma, neg_sigma = AnalysisState.computer_mean_sigma(
            test_data, "batch_prefill"
        )

        # 验证计算结果
        self.assertAlmostEqual(mean[0], 2.0, places=1)
        self.assertAlmostEqual(pos_sigma[0], 3.0, places=1)  # 2 + 1 (标准差)

        # 测试单点数据分支
        test_single_point = {
            State(batch_prefill=1): [1.0]
        }
        x, mean, pos_sigma, neg_sigma = AnalysisState.computer_mean_sigma(
            test_single_point, "batch_prefill"
        )
        self.assertEqual(mean[0], 1.0)
        self.assertEqual(pos_sigma[0], 1.0)
        self.assertEqual(neg_sigma[0], 1.0)
