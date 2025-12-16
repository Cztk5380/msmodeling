# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
训练预测每个状态速度的线性模型
"""
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xgboost
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as MSE

from experimental.data_feature.dataset import MyDataSet
from msguard.security import open_s


class StateXgbModel:
    def __init__(self, train_param: Optional[Dict] = None, update_param: Optional[Dict] = None,
                 save_model_path: Optional[Path] = None,
                 load_model_path: Optional[Path] = None,
                 save_model: bool = True,
                 show_test_data_prediction: bool = False,
                 show_feature_importance: bool = False,

                 ):
        self.train_param = train_param if train_param else {'objective': 'reg:squarederror'}
        self.update_param = update_param if update_param else {'process_type': 'update', 'updater': 'prune'}
        self.save_model_path = save_model_path
        self.load_model_path = load_model_path
        self.show_test_data_prediction = show_test_data_prediction
        self.show_feature_importance = show_feature_importance
        self.save_model = save_model
        if self.save_model and not self.save_model_path:
            raise ValueError("save_model_path can't be empty")

    def train(self, dataset: MyDataSet, train_type: str = 'fit', middle_save_path: Optional[Path] = None):
        """
        训练模型，支持全新训练，和增量训练
        :param dataset: 训练目标数据
        :param train_type: 训练方式，支持全新训练和增量训练。默认default，全新训练，update 为增量训练，更新原来的模型。
        :return:
        """
        logger.debug("train")
        dtrain = xgboost.DMatrix(dataset.train_x, label=dataset.train_y)
        dtest = xgboost.DMatrix(dataset.test_x, label=dataset.test_y)
        if train_type == 'partial_fit':
            model = xgboost.train(self.update_param, dtrain,
                                  xgb_model=self.load_model_path, evals=[(dtest, 'validation')])
        else:
            model = xgboost.train(self.train_param, dtrain, evals=[(dtest, 'validation')])
        y_pred = model.predict(dtest)
        rmse = np.sqrt(MSE(dataset.test_y.values.flatten().tolist(), y_pred))
        if self.show_test_data_prediction:
            plot_pred_and_test(y_pred, dataset, save_path=middle_save_path)
        if self.show_feature_importance:
            plot_feature_importance(model, save_path=middle_save_path)
        if self.save_model:
            model.save_model(self.save_model_path)
        return rmse

    def predict(self, data: np.ndarray) -> np.ndarray:
        _model = xgboost.Booster()
        _model.load_model(self.load_model_path)
        res = _model.predict(xgboost.DMatrix(data, feature_names=_model.feature_names))
        return res


def plot_feature_importance(model, save_path: Optional[Path] = None):
    fig, ax = plt.subplots(figsize=(15, 8))
    logger.debug('weight score %s', model.get_score(importance_type='weight'))
    xgboost.plot_importance(model, ax=ax)
    plt.title('weight score')
    if save_path:
        plt.savefig(save_path.joinpath("weight_score.png"))
        plt.close()
    else:
        plt.show()
    logger.debug('gain score %s', model.get_score(importance_type='gain'))
    fig, ax = plt.subplots(figsize=(15, 8))
    xgboost.plot_importance(model, ax=ax, importance_type='gain')
    plt.title('gain score')
    if save_path:
        plt.savefig(save_path.joinpath("gain_score.png"))
        plt.close()
    else:
        plt.show()
    logger.debug('cover score', model.get_score(importance_type='cover'))
    fig, ax = plt.subplots(figsize=(15, 8))
    xgboost.plot_importance(model, ax=ax, importance_type='cover')
    plt.title('cover score')
    if save_path:
        plt.savefig(save_path.joinpath("cover_score.png"))
        plt.close()
        with open_s(save_path.joinpath("feature_importance.txt"), 'w') as f:
            f.write(f"weight score: {model.get_score(importance_type='weight')} \n")
            f.write(f"gain score: {model.get_score(importance_type='gain')} \n")
            f.write(f"cover score: {model.get_score(importance_type='cover')} \n")
    else:
        plt.show()


def plot_pred_and_test(pred, my_data, save_path: Optional[Path] = None):
    pred_1 = [i for i in pred]
    test_y1 = [i for i in my_data.test_y.values.flatten().tolist()]
    plt.figure()
    plt.scatter(range(len(test_y1)), test_y1, label='test_y', alpha=0.5)
    plt.scatter(range(len(pred_1)), pred_1, label="pred", alpha=0.5)
    plt.title("predict value and test value on train model")
    plt.xlabel("index")
    plt.ylabel("value")
    plt.legend()
    if save_path:
        plt.savefig(save_path.joinpath("predict value and test value on train model.png"))
        plt.close()
    else:
        plt.show()


