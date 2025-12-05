# !/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
"""
所有需要设置的参数
"""
import os
from pathlib import Path
from typing import Dict
from dataclasses import dataclass, field
from msserviceprofiler.modelevalstate.common import StateType
from msserviceprofiler.modelevalstate.inference.constant import OpAlgorithm


@dataclass
class StateParam:
    """
    运行时所有需要的参数
    """
    title: str = "MixModel"
    base_path: Path = Path(os.getcwd())  # 项目的根目录
    model_dir: Path = field(init=False)
    step_dir: Path = field(init=False)
    bak_dir: Path = field(init=False)
    comments: str = "备注"

    xgb_model_train_param: Dict = field(
        default_factory=lambda: {'objective': 'reg:squarederror', })  # xgb model 训练时，需要的模型参数
    xgb_model_show_feature_importance: bool = True  # 是否展示特征重要性
    xgb_model_show_test_data_prediction: bool = True  # 是否展示测试集预测结果
    xgb_model_save_model_path: Path = field(init=False)  # 保存模型的路径
    xgb_model_load_model_path: Path = field(init=False)  # 加载模型的路径, 进行预测使用
    xgb_model_update_param: Dict = field(
        default_factory=lambda: {'process_type': 'update', 'updater': 'prune'})  # 增量训练，更新xgb model时，使用的参数

    start_num_lines: int = 8000  # 第一次读取数据全量训练的行数
    read_num_lines: int = 1000  # 读取数据的时候，一次读取的数据行数，默认1000行
    predict_field: str = "model_execute_time"
    state_type: StateType = StateType.DEFAULT
    plot_velocity_std: bool = True
    plot_predict_std: bool = True
    plot_data_feature: bool = False  # 绘制输出数据的特征和分布
    plot_input_time_with_predict: bool = True
    op_algorithm: OpAlgorithm = OpAlgorithm.EXPECTED

    save_model: bool = True  # 是否保存模型
    shuffle: bool = False
    plot_pred_and_real: bool = True

    def __post_init__(self):
        self.model_dir: Path = self.base_path.joinpath("model")
        self.step_dir: Path = self.base_path.joinpath("step")
        self.bak_dir: Path = self.base_path.joinpath("bak")
        self.model_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
        self.step_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
        self.bak_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
        self.xgb_model_save_model_path: Path = self.model_dir.joinpath("xgb_model.ubj")  # 保存模型的路径
        self.xgb_model_load_model_path: Path = self.model_dir.joinpath("xgb_model.ubj")  # 加载模型的路径, 进行预测使用
