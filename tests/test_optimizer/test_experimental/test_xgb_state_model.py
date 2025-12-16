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

from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest
import numpy as np
import xgboost

from src.optimize.experimental.data_feature.dataset import MyDataSet
from src.optimize.experimental.model.xgb_state_model import StateXgbModel, plot_feature_importance, \
    plot_pred_and_test


# Fixtures
@pytest.fixture
def mock_dataset():
    dataset = MagicMock(spec=MyDataSet)
    dataset.train_x = np.random.rand(100, 5)
    dataset.train_y = MagicMock()
    dataset.train_y.values = np.random.rand(100, 1)
    dataset.test_x = np.random.rand(20, 5)
    dataset.test_y = MagicMock()
    dataset.test_y.values = np.random.rand(20, 1)
    return dataset


@pytest.fixture
def tmp_path(tmpdir):
    return Path(tmpdir)


# Test Cases
class TestStateXgbModel():
    @staticmethod
    def test_init_without_save_path():
        with pytest.raises(ValueError, match="save_model_path can't be empty"):
            StateXgbModel(save_model=True, save_model_path=None)

    @staticmethod
    def test_init_with_save_path(tmp_path):  # 修复了属性名错误
        model = StateXgbModel(save_model=True, save_model_path=tmp_path)
        assert model.save_model_path == tmp_path  # 修复了属性名拼写错误

    @staticmethod
    def test_init_without_train_visualization():
        model = StateXgbModel(save_model=False)
        assert model.show_test_data_prediction is False
        assert model.show_feature_importance is False

    @staticmethod
    @patch('experimental.model.xgb_state_model.xgboost.Booster')
    @patch('experimental.model.xgb_state_model.xgboost.DMatrix')
    def test_predict(mock_dmatrix, mock_booster, tmp_path):
        # 创建模拟的Booster实例
        mock_model = MagicMock()
        mock_model.feature_names = ['f1', 'f2', 'f3']
        mock_model.predict.return_value = np.array([0.5, 0.6])

        # 设置Booster构造函数返回模拟模型
        mock_booster.return_value = mock_model

        # 创建模拟的DMatrix
        mock_dmatrix_instance = MagicMock()
        mock_dmatrix_instance.feature_names = ['f1', 'f2', 'f3']
        mock_dmatrix.return_value = mock_dmatrix_instance

        # 创建模型对象
        model = StateXgbModel(
            load_model_path=tmp_path / "model.bin",
            save_model=False
        )

        # 模拟test data
        test_data = np.random.rand(2, 3)

        # 执行测试
        result = model.predict(test_data)

        # 验证预测结果
        assert result.tolist() == [0.5, 0.6]

        # 验证方法调用
        mock_booster.assert_called_once()  # 验证Booster被实例化
        mock_model.load_model.assert_called_once_with(tmp_path / "model.bin")
        mock_dmatrix.assert_called_once_with(test_data, feature_names=['f1', 'f2', 'f3'])
        mock_model.predict.assert_called_once_with(mock_dmatrix_instance)

    @staticmethod
    @patch('experimental.model.xgb_state_model.plt.show')
    @patch('experimental.model.xgb_state_model.plt.close')
    @patch('experimental.model.xgb_state_model.plt.savefig')
    @patch('experimental.model.xgb_state_model.plt.subplots')
    @patch('builtins.open', MagicMock())
    @patch('experimental.model.xgb_state_model.xgboost.plot_importance')
    def test_plot_feature_importance(mock_plot, mock_subplots, mock_savefig, mock_close, mock_show, tmp_path):
        # 设置mock模型
        mock_model = MagicMock()
        mock_model.get_score.return_value = {'feature1': 1.0, 'feature2': 0.5}

        # Mock图表返回
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Test with save path
        plot_feature_importance(mock_model, save_path=tmp_path)

        # 验证保存图片被调用
        save_calls = [call.args[0] for call in mock_savefig.mock_calls if hasattr(call, 'args')]
        assert any("weight_score.png" in str(call) for call in save_calls)
        assert any("gain_score.png" in str(call) for call in save_calls)
        assert any("cover_score.png" in str(call) for call in save_calls)


    @staticmethod
    @patch('src.optimize.experimental.model.xgb_state_model.xgboost.train')
    @patch('src.optimize.experimental.model.xgb_state_model.xgboost.DMatrix')
    @patch('src.optimize.experimental.model.xgb_state_model.plot_feature_importance')
    @patch('src.optimize.experimental.model.xgb_state_model.plot_pred_and_test')
    def test_train_with_visualization(mock_plot_pred, mock_plot_feat, mock_dmatrix, mock_train, mock_dataset,
                                      tmp_path):
        # 设置mock返回
        mock_model = MagicMock(spec=xgboost.Booster)

        # 设置预测返回值
        mock_model.predict.return_value = np.random.rand(20)
        mock_model.save_model = MagicMock()  # 添加保存方法

        mock_train.return_value = mock_model
        mock_dmatrix.return_value = MagicMock()

        # 模拟模型
        model = StateXgbModel(
            save_model_path=tmp_path / "model.bin",
            show_test_data_prediction=True,
            show_feature_importance=True
        )

        # 训练模型
        rmse = model.train(mock_dataset, middle_save_path=tmp_path)

        # 验证结果
        assert isinstance(rmse, float)
        mock_plot_feat.assert_called_once()
        mock_plot_pred.assert_called_once()

        # 验证保存方法被调用
        mock_model.save_model.assert_called_once()

    @staticmethod
    @patch('experimental.model.xgb_state_model.xgboost.train')
    @patch('experimental.model.xgb_state_model.xgboost.DMatrix')
    @patch('experimental.model.xgb_state_model.plot_feature_importance')
    @patch('experimental.model.xgb_state_model.plot_pred_and_test')
    def test_train_without_visualization(mock_plot_pred, mock_plot_feat, mock_dmatrix, mock_train, mock_dataset,
                                         tmp_path):
        # 设置mock返回
        mock_model = MagicMock(spec=xgboost.Booster)

        # 设置预测返回值
        mock_model.predict.return_value = np.random.rand(20)
        mock_model.save_model = MagicMock()  # 添加保存方法

        mock_train.return_value = mock_model
        mock_dmatrix.return_value = MagicMock()

        # 模拟模型
        model = StateXgbModel(
            save_model_path=tmp_path / "model.bin",
            show_test_data_prediction=False,
            show_feature_importance=False
        )

        # 训练模型
        rmse = model.train(mock_dataset)

        # 验证结果
        assert isinstance(rmse, float)
        mock_plot_feat.assert_not_called()
        mock_plot_pred.assert_not_called()

        # 验证保存方法被调用
        mock_model.save_model.assert_called_once()

    @staticmethod
    @patch('experimental.model.xgb_state_model.xgboost.train')
    @patch('experimental.model.xgb_state_model.xgboost.DMatrix')
    def test_train_without_save(mock_dmatrix, mock_train, mock_dataset):
        # 设置mock返回
        mock_model = MagicMock(spec=xgboost.Booster)
        mock_model.predict.return_value = np.random.rand(20)
        mock_model.save_model = MagicMock()  # 添加保存方法

        mock_train.return_value = mock_model
        mock_dmatrix.return_value = MagicMock()

        model = StateXgbModel(
            save_model=False
        )
        rmse = model.train(mock_dataset)

        assert isinstance(rmse, float)
        mock_train.assert_called_once()

        # 验证没有调用保存方法
        mock_model.save_model.assert_not_called()

    @staticmethod
    @patch('experimental.model.xgb_state_model.plt.show')
    @patch('experimental.model.xgb_state_model.plt.close')
    @patch('experimental.model.xgb_state_model.plt.savefig')
    @patch('experimental.model.xgb_state_model.plt.figure')
    @patch('experimental.model.xgb_state_model.plt.scatter')
    @patch('experimental.model.xgb_state_model.plt.title')
    @patch('experimental.model.xgb_state_model.plt.xlabel')
    @patch('experimental.model.xgb_state_model.plt.ylabel')
    @patch('experimental.model.xgb_state_model.plt.legend')
    def test_plot_pred_and_test(mock_legend, mock_ylabel, mock_xlabel, mock_title, mock_scatter, mock_figure,
                                mock_savefig, mock_close, mock_show, tmp_path):
        # 创建模拟数据
        pred = np.array([0.1, 0.2, 0.3, 0.4])
        mock_dataset = MagicMock()
        mock_dataset.test_y = MagicMock()
        mock_dataset.test_y.values = np.array([[0.15], [0.25], [0.35], [0.45]])

        # 测试保存图片的情况
        plot_pred_and_test(pred, mock_dataset, save_path=tmp_path)

        # 验证函数调用
        mock_figure.assert_called_once()
        mock_scatter.assert_any_call(range(4), [0.15, 0.25, 0.35, 0.45], label='test_y', alpha=0.5)
        mock_scatter.assert_any_call(range(4), [0.1, 0.2, 0.3, 0.4], label="pred", alpha=0.5)
        mock_title.assert_called_once_with("predict value and test value on train model")
        mock_xlabel.assert_called_once_with("index")
        mock_ylabel.assert_called_once_with("value")
        mock_legend.assert_called_once()
        # 修复文件名中的空格问题
        mock_savefig.assert_called_once_with(tmp_path.joinpath("predict value and test value on train model.png"))
        mock_close.assert_called_once()
        mock_show.assert_not_called()

        # 重置mock
        mock_figure.reset_mock()
        mock_scatter.reset_mock()
        mock_title.reset_mock()
        mock_xlabel.reset_mock()
        mock_ylabel.reset_mock()
        mock_legend.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        mock_show.reset_mock()

        # 测试显示图片的情况
        plot_pred_and_test(pred, mock_dataset, save_path=None)

        # 验证函数调用
        mock_figure.assert_called_once()
        mock_scatter.assert_any_call(range(4), [0.15, 0.25, 0.35, 0.45], label='test_y', alpha=0.5)
        mock_scatter.assert_any_call(range(4), [0.1, 0.2, 0.3, 0.4], label="pred", alpha=0.5)
        mock_title.assert_called_once_with("predict value and test value on train model")
        mock_xlabel.assert_called_once_with("index")
        mock_ylabel.assert_called_once_with("value")
        mock_legend.assert_called_once()
        mock_savefig.assert_not_called()
        mock_close.assert_not_called()
        mock_show.assert_called_once()
