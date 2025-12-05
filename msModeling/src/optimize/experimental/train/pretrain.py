# !/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
"""
1. 训练基础模型
2. 用基础模型拟合概率
3. 用新增加数据继续训练模型
"""
import argparse
import glob
import shutil
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Tuple, Optional, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from pandas import DataFrame
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from msserviceprofiler.msguard import Rule
from msserviceprofiler.modelevalstate.analysis import AnalysisState
from msserviceprofiler.modelevalstate.common import _DECODE, _PREFILL, State
from msserviceprofiler.modelevalstate.common import computer_speed_with_second, get_train_sub_path, \
    update_global_coefficient

try:
    from msserviceprofiler.modelevalstate.data_feature.dataset_with_modin import MyDataSetWithModin as MyDataSet
except ModuleNotFoundError:
    try:
        from msserviceprofiler.modelevalstate.data_feature.dataset_with_swifter import MyDataSetWithSwifter as MyDataSet
    except ModuleNotFoundError:
        from msserviceprofiler.modelevalstate.data_feature.dataset import MyDataSet
from msserviceprofiler.modelevalstate.data_feature.dataset import CustomOneHotEncoder, CustomLabelEncoder, \
    preset_category_data
from msserviceprofiler.modelevalstate.data_feature.v1 import FileReader
from msserviceprofiler.modelevalstate.inference.common import HistInfo, model_op_size, OP_EXPECTED_FIELD_MAPPING, \
    OP_SCALE_HIST_FIELD_MAPPING
from msserviceprofiler.modelevalstate.inference.constant import OpAlgorithm
from msserviceprofiler.modelevalstate.model.xgb_state_model import StateXgbModel
from msserviceprofiler.modelevalstate.train.state_param import StateParam
from msserviceprofiler.modelevalstate.inference.utils import save_dataframe_to_csv


@dataclass
class NodeInfo:
    stage: str  # 当前模型状态的类型 Prefill/Decode
    batch_size: int  # 当前状态处理的请求个数


class PretrainModel:

    def __init__(self, state_param: Optional[StateParam] = None, dataset: Optional[MyDataSet] = None,
                 model: Optional[StateXgbModel] = None, plt_data: bool = False):
        self.state_param = state_param
        self.dataset = dataset
        self.model = model
        self.plt_data = plt_data
        self.computer_up_expectations = []
        self.computer_ud_expectations = []
        self.real_up_expectations = []
        self.real_ud_expectations = []
        self.rmse = []
        self.r2 = []
        self.mape = []

    @staticmethod
    def get_decode_and_prefill_time(target: Tuple[NodeInfo], field: str):
        # 获取不同类型batch的运行时间
        _inner_prefill_time = {}
        _inner_decode_time = {}
        for line in target:
            _cur_state = State()
            if line.stage == _PREFILL:
                _cur_state.batch_prefill = line.batch_size
                update_global_coefficient(_inner_prefill_time, _cur_state, getattr(line, field))
            elif line.stage == _DECODE:
                _cur_state.batch_decode = line.batch_size
                update_global_coefficient(_inner_decode_time, _cur_state, getattr(line, field))
        return _inner_prefill_time, _inner_decode_time

    @staticmethod
    def get_up_ud(target: Tuple[NodeInfo], field: str):
        # 获取不同类型的batch运行的单位时延迟
        _inner_up = {}
        _inner_ud = {}
        for line in target:
            _cur_state = State()
            if line.stage == _PREFILL:
                _cur_state.batch_prefill = line.batch_size
                update_global_coefficient(_inner_up, _cur_state, computer_speed_with_second(line, field))
            elif line.stage == _DECODE:
                _cur_state.batch_decode = line.batch_size
                update_global_coefficient(_inner_ud, _cur_state, computer_speed_with_second(line, field))
        return _inner_up, _inner_ud

    @staticmethod
    def get_stage_after_preprocess(row: pd.Series, encoder: Union[CustomOneHotEncoder, CustomLabelEncoder]):
        # 根据预处理后的数据识别该行数据是decode 还是prefill
        if isinstance(encoder) == CustomOneHotEncoder:
            batch_stage_encoder = \
                [encoder.one_hot_encoders[i] for i, v in enumerate(encoder.one_hots) if v.name == "batch_stage"][0]
            _batch_index = [i for i in row.index if "batch_stage" in i]
            stage = batch_stage_encoder.inverse_transform([[int(row[i]) for i in _batch_index]])

        else:
            batch_stage_encoder = \
                [encoder.category_encoders[i] for i, v in enumerate(encoder.category_info) if v.name == "batch_stage"][
                    0]
            stage = batch_stage_encoder.inverse_transform([int(row.batch_stage)])[0]

        return stage

    @staticmethod
    def get_nodes_with_origin_data(features: DataFrame, labels: DataFrame, predict_field: str,
                                   encoder: Union[CustomOneHotEncoder, CustomLabelEncoder]):
        # 获取原来的node信息
        target_data = []
        for ind, row in features.iterrows():
            stage = PretrainModel.get_stage_after_preprocess(row, encoder)
            _cur_node = NodeInfo(stage, row.batch_size)
            setattr(_cur_node, predict_field, labels.iloc[ind, 0])
            target_data.append(_cur_node)
        return tuple(target_data)

    def train(self, lines_data: Optional[DataFrame] = None,
              middle_save_path: Optional[Path] = None):
        logger.info("start train")
        self.dataset.construct_data(lines_data, plt_data=self.plt_data, middle_save_path=middle_save_path)
        rmse = self.model.train(self.dataset, middle_save_path=middle_save_path)
        logger.debug(f"rmse {rmse}")

        self.rmse.append(rmse)

    def partial_train(self, lines_data: Optional[DataFrame] = None,
                      middle_save_path: Optional[Path] = None):
        logger.info("start partial train")
        self.dataset.custom_encoder.fit(load=True)
        self.dataset.construct_data(lines_data, plt_data=self.plt_data, middle_save_path=middle_save_path)
        rmse = self.model.train(self.dataset, train_type="partial_fit", middle_save_path=middle_save_path)
        self.rmse.append(rmse)

    def get_nodes_with_model_predict(self, features: DataFrame):
        # 使用模型进行预测
        target_data = []
        for _, row in features.iterrows():
            try:
                _predict = self.model.predict((row,))[0]
                stage = self.get_stage_after_preprocess(row, self.dataset.custom_encoder)
                _cur_node = NodeInfo(stage, row.batch_size)
                setattr(_cur_node, self.state_param.predict_field, _predict)
                target_data.append(_cur_node)
            except Exception as e:
                logger.error(f"get_nodes_with_model_predict时出错：{e}")  
        return tuple(target_data)

    def predict_and_plot(self, features: DataFrame, labels: DataFrame, predict_field: str, save_path: Optional[Path]):
        origin_data = self.get_nodes_with_origin_data(features, labels, predict_field, self.dataset.custom_encoder)
        data = self.get_nodes_with_model_predict(features)
        r2 = r2_score([getattr(k, predict_field) for k in origin_data], [getattr(k, predict_field) for k in data])
        self.r2.append(r2)
        mape = mean_absolute_percentage_error([getattr(k, predict_field) for k in origin_data],
                                              [getattr(k, predict_field) for k in data])
        self.mape.append(mape)
        all_up, all_ud = self.get_up_ud(data, predict_field)
        origin_up, origin_ud = self.get_up_ud(tuple(origin_data), predict_field)

        if self.state_param.plot_velocity_std:
            self.plot_velocity_std(origin_up, all_up, origin_ud, all_ud, save_path=save_path)
        if self.state_param.plot_input_time_with_predict:
            # 绘制时间
            _all_prefill_time, _all_decode_time = self.get_decode_and_prefill_time(data, predict_field)
            origin_prefill_time, origin_decode_time = self.get_decode_and_prefill_time(tuple(origin_data),
                                                                                       predict_field)
            AnalysisState.plot_input_velocity_with_predict(origin_prefill_time, _all_prefill_time, "batch_prefill",
                                                           f"origin and predict prefill time {predict_field} std",
                                                           "batch_prefill",
                                                           "time us", save_path=save_path)
            AnalysisState.plot_input_velocity_with_predict(origin_decode_time, _all_decode_time, "batch_decode",
                                                           f"origin and predict decode time {predict_field} std",
                                                           "batch_decode",
                                                           "time us", save_path=save_path)
        return all_up, all_ud

    def predict_and_plot_with_speed(self, features: DataFrame, labels: DataFrame, save_path: Optional[Path]):
        logger.debug("predict test data.")
        _predicts = self.model.predict(features.values)
        _origin_data = labels
        r2 = r2_score(labels, _predicts)
        self.r2.append(r2)
        logger.debug(f"r2: {r2}")
        mape = mean_absolute_percentage_error(labels, _predicts)
        self.mape.append(mape)
        logger.debug(f"mape: {mape}")
        _predict_df = pd.DataFrame({"batch_stage": self.dataset.load_data["batch_stage"],
                                    "batch_size": self.dataset.load_data["batch_size"],
                                    "predict": _predicts})
        _origin_df = pd.DataFrame({"batch_stage": self.dataset.load_data["batch_stage"],
                                   "batch_size": self.dataset.load_data["batch_size"],
                                   "predict": _predicts})
        if self.state_param.plot_input_time_with_predict:
            AnalysisState.plot_input_velocity_with_df(_predict_df, _origin_df, save_path)

    def predict(self, lines_data: DataFrame,
                save_path: Optional[Path] = None):
        logger.info("start predict")
        self.dataset.construct_data(lines_data, plt_data=self.plt_data, middle_save_path=save_path)
        try:
            return self.predict_and_plot_with_speed(self.dataset.features, self.dataset.labels, save_path)
        except (AttributeError, OverflowError, KeyError, ValueError, RuntimeError):
            return self.predict_and_plot(self.dataset.features, self.dataset.labels, self.state_param.predict_field,
                                         save_path=save_path)

    def plot_velocity_std(self, origin_up, all_up, origin_ud, all_ud, save_path: Optional[Path] = None):
        logger.info("start plot velocity std")
        # 对比Up,Ud的分布
        AnalysisState.plot_input_velocity_with_predict(origin_up, all_up, "batch_prefill",
                                                       f"origin and predict up {self.state_param.predict_field} std",
                                                       "batch_prefill",
                                                       "velocity", save_path=save_path)
        AnalysisState.plot_input_velocity_with_predict(origin_ud, all_ud, "batch_decode",
                                                       f"origin and predict ud {self.state_param.predict_field} std",
                                                       "batch_decode",
                                                       "velocity", save_path=save_path)
        AnalysisState.plot_input_velocity(origin_up, "batch_prefill", f"up {self.state_param.predict_field} std",
                                          "batch_prefill",
                                          "velocity", save_path=save_path)
        AnalysisState.plot_input_velocity(origin_ud, "batch_decode", f"ud {self.state_param.predict_field} std",
                                          "batch_decode",
                                          "velocity", save_path=save_path)

    def bak_model(self, increment_stage: str = "base"):
        _bak_dir = self.state_param.bak_dir.joinpath(increment_stage)
        _bak_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
        shutil.copy(self.state_param.xgb_model_save_model_path,
                    _bak_dir.joinpath(self.state_param.xgb_model_save_model_path.name))

    def plot_metric(self, save_path: Optional[Path] = None):
        data = {"rmse": self.rmse, "r2": self.r2, "mape": self.mape}
        df = pd.DataFrame(data)
        sns.scatterplot(df)
        if save_path:
            plt.savefig(save_path.joinpath("metric.png"))
            plt.close()
        else:
            plt.show()


class TrainVersion1:
    @staticmethod 
    def simple_train(file_paths: List[Path], sp: StateParam, pm: PretrainModel):
        # 训练模型，将全部数据1:9分，9进行训练，1进行预测。
        fl = FileReader(file_paths)
        line_data = fl.read_lines()
        line_data = line_data.dropna()

        def replace_none(value):
            if isinstance(value, str) and 'None' in value:
                return None
            return value
        
        line_data = line_data.applymap(replace_none)
        line_data = line_data.dropna()
        train_data, test_data = train_test_split(line_data, test_size=0.1, shuffle=True)
        logger.debug(f"train data shape {train_data.shape}")
        sp.comments = f"input files: {file_paths} \n"
        save_path = sp.step_dir.joinpath("base")
        save_path.mkdir(parents=True, exist_ok=True, mode=0o750)
        pm.train(train_data.reset_index(drop=True), middle_save_path=save_path)
        sp.comments += f'feature shape {pm.dataset.features.shape}\n'
        sp.comments += (f"data shuffle: True, \n train case: {pm.dataset.train_x.shape}, "
                        f"validate case: {pm.dataset.test_x.shape}, predict case: {test_data.shape} \n")
        pm.bak_model()
        logger.debug("test data {test_data.shape}")
        save_path = sp.step_dir.joinpath("1")
        save_path.mkdir(parents=True, exist_ok=True, mode=0o750)
        pm.predict(test_data.reset_index(drop=True), save_path)
        logger.info("finished train")


parser = argparse.ArgumentParser(prog="Train Model.")
parser.add_argument("-i", "--input", default=None, type=Path, required=True)
parser.add_argument("-o", "--output", default=Path("output"), type=Path)


def pretrain(input_path, output_path):
    _input_file = Path(input_path).expanduser().resolve()
    if not Rule.input_dir_traverse.is_satisfied_by(_input_file):
        logger.error("not found dir for train model")
        return
    # 获取目录中所有的feature.csv
    train_files = glob.glob(f"{_input_file}/**/*feature.csv", recursive=True)
    if not train_files:
        raise FileNotFoundError(f"{_input_file}/**/*feature.csv")
    train_files = [Path(p) for p in train_files]
    # 创建输出目录
    output = Path(output_path).expanduser().resolve()
    if not output.exists:
        output.mkdir(parents=True, mode=0o750)
    # 运行模型训练
    sp = StateParam(
        base_path=output,
        predict_field="model_execute_time",
        save_model=True,
        shuffle=True,
        plot_pred_and_real=False,
        plot_data_feature=False,
        plot_velocity_std=False,
        plot_predict_std=False,
        op_algorithm=OpAlgorithm.EXPECTED,
        xgb_model_show_test_data_prediction=False,
        xgb_model_show_feature_importance=False,
        plot_input_time_with_predict=False,
        title="MixModel without warmup with service info"
    )
    model = StateXgbModel(
        train_param=sp.xgb_model_train_param,
        update_param=sp.xgb_model_update_param,
        save_model_path=sp.xgb_model_save_model_path,
        load_model_path=sp.xgb_model_save_model_path,
        show_test_data_prediction=sp.xgb_model_show_test_data_prediction,
        show_feature_importance=sp.xgb_model_show_feature_importance,
    )
    custom_encoder = CustomLabelEncoder(preset_category_data)
    custom_encoder.fit()
    dataset = MyDataSet(custom_encoder=custom_encoder, predict_field=sp.predict_field,
                        shuffle=sp.shuffle, op_algorithm=sp.op_algorithm)
    pm = PretrainModel(state_param=sp, dataset=dataset, model=model, plt_data=sp.plot_data_feature)
    TrainVersion1.simple_train(train_files, sp, pm)
    train_data = dataset.features.copy(deep=False)
    train_data["label"] = dataset.labels
    _train_dir = output.joinpath("cache")
    save_dataframe_to_csv(train_data, _train_dir, "train_data.csv")


def main(args):
    # 解析命令
    _input_file = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    pretrain(_input_file, output)


