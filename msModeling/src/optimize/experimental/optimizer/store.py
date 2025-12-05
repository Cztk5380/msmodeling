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
import csv
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from msserviceprofiler.msguard.security import open_s, sanitize_csv_value
from msserviceprofiler.modelevalstate.config.config import (
    DataStorageConfig,
    RUN_TIME,
    PerformanceIndex,
    OptimizerConfigField,
    get_settings
)
from msserviceprofiler.modelevalstate.optimizer.plugins.benchmark import VllmBenchMark, AisBench
from msserviceprofiler.modelevalstate.optimizer.plugins.simulate import Simulator, VllmSimulator
from msserviceprofiler.modelevalstate.common import read_csv_s


LLM_MODEL = "llm_model"
DATASET_PATH = "dataset_path"
SIMULATOR = "simulator"
NUM_PROMPTS = "num_prompts"
MAX_OUTPUT_LEN = "max_output_len"


class DataStorage:
    def __init__(self, config: DataStorageConfig, simulator=None, benchmark=None, ):
        self.config = config
        if not self.config.store_dir.exists():
            self.config.store_dir.mkdir(parents=True, mode=0o750)
        self.save_file = self.config.store_dir.joinpath(f"data_storage_{RUN_TIME}.csv")
        self.simulator = simulator
        self.benchmark = benchmark

    @staticmethod
    def load_history_position(load_dir: Path, filter_field: Optional[Dict] = None) -> Optional[List]:
        if not load_dir.exists():
            raise FileNotFoundError(f"file: {load_dir}")
        if not load_dir.is_dir():
            raise ValueError(f"Expect a directory, not a file.")
        history_data = []
        for file in sorted([f for f in load_dir.iterdir() if f.is_file()], key=lambda x: x.stat().st_ctime):
            if file.name.startswith("data_storage") and file.suffix == ".csv":
                data = read_csv_s(file).to_dict(orient="records")
                history_data.extend(data)
        if not history_data:
            return None
        return DataStorage.filter_data(history_data, filter_field)

    @staticmethod
    def filter_data(datas: List[Dict], filter_field: Optional[Dict] = None):
        if not filter_field:
            return datas
        filter_datas = []
        for d in datas:
            flag = False
            for k, v in filter_field.items():
                # 不存在的字段 无法进行筛选
                if k not in d:
                    continue
                # 字段存在 但不等于目标字段，则去掉
                if d[k].strip().lower() != v.strip().lower():
                    flag = True
                    break
            if flag:
                continue
            else:
                filter_datas.append(d)
        return filter_datas

    def get_run_info(self):
        _run_info = {}
        if self.benchmark is None:
            return _run_info
        if isinstance(self.benchmark, (AisBench, VllmBenchMark)):
            _run_info[NUM_PROMPTS] = self.benchmark.config.command.num_prompts
        elif self.benchmark.num_prompts:
            _run_info[NUM_PROMPTS] = self.benchmark.num_prompts
        return _run_info

    def save(self, performance_index: PerformanceIndex, params: Tuple[OptimizerConfigField], **kwargs):
        logger.info(f"Save result with DataStorage. File path: {self.save_file!r}")
        _column = []
        _value = []
        for k, v in performance_index.model_dump().items():
            _column.append(k)
            _value.append(v)
        for _p in params:
            _column.append(_p.name)
            _value.append(_p.value)
        for k, v in kwargs.items():
            _column.append(k)
            _value.append(v)
        for k, v in self.get_run_info().items():
            _column.append(k)
            _value.append(v)
        if self.save_file.exists():
            with open_s(self.save_file, "a+") as f:
                data_writer = csv.writer(f)
                data_writer.writerow([sanitize_csv_value(_v) for _v in _value])
        else:
            with open_s(self.save_file, "w") as f:
                data_writer = csv.writer(f)
                data_writer.writerow(_column)
                data_writer.writerow([sanitize_csv_value(_v) for _v in _value])

    def get_best_result(self):
        settings = get_settings()
        optimizer_result = read_csv_s(self.save_file)
        optimizer_result = optimizer_result.replace([np.inf, -np.inf], np.nan)
        pso_result = optimizer_result
        if self.benchmark:
            # 提取公共属性访问	
            command = self.benchmark.config.command	
            request_nums = command.num_prompts	
            pso_result = optimizer_result[optimizer_result[NUM_PROMPTS] == request_nums]
        pso_result = pso_result.dropna(subset="fitness")
        pso_result = pso_result[pso_result["time_to_first_token"] > 0]
        pso_result = pso_result[pso_result["time_per_output_token"] > 0]
        pso_result = pso_result[pso_result["generate_speed"] > 0]
        pso_result = pso_result.reset_index()
        _fitness_index = pso_result.nsmallest(self.config.pso_top_k, "fitness").index
        if settings.ttft_penalty and settings.tpot_penalty:
            _generate_speed_index = pso_result[
                (pso_result["time_to_first_token"] <= settings.ttft_slo * (1 + settings.slo_coefficient)) &
                (pso_result["time_per_output_token"] <= settings.tpot_slo * (1 + settings.slo_coefficient))].nlargest(
                self.config.pso_top_k, "generate_speed").index
        elif settings.tpot_penalty:
            _generate_speed_index = pso_result[
                pso_result["time_per_output_token"] <= settings.tpot_slo * (1 + settings.slo_coefficient)].nlargest(
                self.config.pso_top_k, "generate_speed").index
        else:
            _generate_speed_index = pso_result.nlargest(self.config.pso_top_k, "generate_speed").index
        _fine_tune_index = _fitness_index.union(_generate_speed_index)
        return pso_result.iloc[_fine_tune_index]