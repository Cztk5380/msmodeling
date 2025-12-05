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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

from loguru import logger

from msserviceprofiler.modelevalstate.inference.data_format_v1 import EnvField, HardWare, MindieConfig, \
    ModelConfig, ModelOpField, ModelStruct, \
    ENV_FIELD, HARDWARE_FIELD, MINDIE_FIELD, MODEL_CONFIG_FIELD, MODEL_STRUCT_FIELD, MODEL_OP_FIELD, BATCH_SIZE, \
    MAX_SEQ_LEN
from msserviceprofiler.msguard.security import open_s


@dataclass
class StaticFile:
    base_path: Path = Path("data/model")
    hardware_path: Optional[Path] = None
    env_path: Optional[Path] = None
    mindie_config_path: Optional[Path] = None
    config_path: Optional[Path] = None
    model_struct_path: Optional[Path] = None
    model_decode_op_path: Optional[Path] = None
    model_prefill_op_path: Optional[Path] = None

    def __post_init__(self):
        if not self.base_path.exists():
            raise FileNotFoundError(self.base_path)
        if self.hardware_path is None:
            self.hardware_path = self.base_path.joinpath("hardware.json")
        if self.env_path is None:
            self.env_path = self.base_path.joinpath("env.json")
        if self.mindie_config_path is None:
            self.mindie_config_path = self.base_path.joinpath("mindie_config.json")
        if self.config_path is None:
            self.config_path = self.base_path.joinpath("model_config.json")
        if self.model_struct_path is None:
            self.model_struct_path = self.base_path.joinpath("model_struct.csv")
        if self.model_decode_op_path is None:
            self.model_decode_op_path = self.base_path.joinpath("model_decode_op.csv")
        if self.model_prefill_op_path is None:
            self.model_prefill_op_path = self.base_path.joinpath("model_prefill_op.csv")
        for path in [self.hardware_path, self.env_path, self.mindie_config_path, self.config_path,
                     self.model_struct_path, self.model_decode_op_path, self.model_prefill_op_path]:
            if not path.exists():
                logger.debug(f"Not Found {path!r}")


class FileHanlder:
    """
    加载静态数据
    """

    def __init__(self, data_file_paths: StaticFile):
        self.data_file_paths = data_file_paths
        self.hardware: Optional[HardWare] = None
        self.env_info: Optional[EnvField] = None
        self.mindie_info: Optional[MindieConfig] = None
        self.model_config_info: Optional[ModelConfig] = None
        self.model_struct_info: Optional[ModelStruct] = None
        self.prefill_op_data: Optional[Dict[Union[int, Tuple], List[ModelOpField]]] = None
        self.decode_op_data: Optional[Dict[Union[int, Tuple], List[ModelOpField]]] = None

    @staticmethod
    def load_hardware_data(hardware_path: Path) -> HardWare:
        with open_s(hardware_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as err:
                logger.warning("Failed to open %r, error: %r" % (hardware_path, err))
                data = {}
        if not data:
            raise AssertionError("Data is None")
        new_data = {}
        for k, v in data.items():
            if isinstance(v, list):
                new_data[k] = tuple(v)
            else:
                new_data[k] = v
        return HardWare(**{k: v for k, v in new_data.items() if k in HARDWARE_FIELD})

    @staticmethod
    def load_env_data(env_path: Path) -> EnvField:
        with open_s(env_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as err:
                logger.warning("Failed to open %r, error: %r" % (env_path, err))
                data = {}
        if not data:
            raise AssertionError("Data is None")
        return EnvField(**{k: v for k, v in data.items() if k in ENV_FIELD})

    @staticmethod
    def load_mindie_config(mindie_config_path: Path) -> MindieConfig:
        with open_s(mindie_config_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as err:
                logger.warning("Failed to open %r, error: %r" % (mindie_config_path, err))
                data = {}
        if not data:
            raise AssertionError("Data is None")
        if "max_seq_len" in data:
            data["mindie__max_seq_len"] = data["max_seq_len"]
        new_data = {}
        for k, v in data.items():
            if isinstance(v, list):
                new_data[k] = tuple(v)
            else:
                new_data[k] = v
        return MindieConfig(**{k: v for k, v in new_data.items() if k in MINDIE_FIELD})

    @staticmethod
    def load_model_config(config_path: Path) -> ModelConfig:
        with open_s(config_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as err:
                logger.warning("Failed to open %r, error: %r" % (config_path, err))
                data = {}
        if not data:
            raise AssertionError("Data is None")
        new_data = {}
        for k, v in data.items():
            if isinstance(v, list):
                new_data[k] = tuple(v)
            else:
                new_data[k] = v
        return ModelConfig(**{k: v for k, v in new_data.items() if k in MODEL_CONFIG_FIELD})

    @staticmethod
    def load_model_struct(model_struct_path: Path) -> ModelStruct:
        _load_field = None
        with open_s(model_struct_path, "r", encoding="utf-8", newline="") as f:
            model_struct_reader = csv.reader(f)
            model_struct = None
            for i, row in enumerate(model_struct_reader):

                if i == 0:
                    _load_field = row
                    if len(row) != len(MODEL_STRUCT_FIELD):
                        raise ValueError(f"Load length {len(row)}, expected length {len(MODEL_STRUCT_FIELD)}")
                    _check = [k for k in MODEL_STRUCT_FIELD if k in row]
                    if tuple(_check) != MODEL_STRUCT_FIELD:
                        raise ValueError(f"Missing Field: {set(MODEL_STRUCT_FIELD) - set(_check)}")
                    continue
                _row = [row[_load_field.index(k)] for k in MODEL_STRUCT_FIELD]
                model_struct = ModelStruct(*_row)
        if not model_struct:
            raise ValueError("model_struct is None")
        return model_struct

    @staticmethod
    def check_filed(row: List[str], op_type: Optional[str] = None) -> str:
        _tmp_row = row
        if BATCH_SIZE in row:
            _tmp_row.remove(BATCH_SIZE)
            op_type = BATCH_SIZE
        if MAX_SEQ_LEN in row:
            _tmp_row.remove(MAX_SEQ_LEN)
            op_type = MAX_SEQ_LEN
        if tuple(_tmp_row) != MODEL_OP_FIELD:
            raise AssertionError(f"get fields: {row}, expected fields: {MODEL_OP_FIELD}")
        return op_type

    @staticmethod
    def process_op_data(row: List[str], op_type: Optional[str], op_path: str, all_op_data: dict, i: int) -> \
        Dict[Tuple, Tuple[ModelOpField]]:
        try:
            for _row in row:
                if not _row:
                    raise ValueError(f"Empty data found in {op_path!r}. i: {i}, row: {row}")
            if op_type == BATCH_SIZE:
                if len(row) < 1:
                    raise ValueError(f"Insufficient data in row {i}. Expected at least 1 column.")
                _relation_key = (int(row[0]),)
                _relation_value = tuple(row[1:])
            else:
                if len(row) < 2:
                    raise ValueError(f"Insufficient data in row {i}. Expected at least 2 columns.")
                _relation_key = (int(row[0]), int(row[1]))
                _relation_value = tuple(row[2:])
            if _relation_key not in all_op_data:
                all_op_data[_relation_key] = (_relation_value,)
            else:
                all_op_data[_relation_key] = (*all_op_data[_relation_key], _relation_value)
        except Exception as e:
            logger.error(f"Unexpected error occurred at row {i}: {e}")
            return None
        return all_op_data

    @staticmethod
    def load_op_data(op_path: Path) -> Dict[Tuple, Tuple[ModelOpField]]:
        op_type = BATCH_SIZE
        all_op_data = {}
        with open_s(op_path, "r", encoding="utf-8", newline="") as f:
            op_reader = csv.reader(f)
            for i, row in enumerate(op_reader):
                if i == 0:
                    op_type = FileHanlder.check_filed(row, op_type)
                    continue
                all_op_data = FileHanlder.process_op_data(row, op_type, op_path, all_op_data, i)
        if not all_op_data:
            raise ValueError("all_op_data is None.")
        return all_op_data

    @staticmethod
    def get_op_field(batch_stage: str, batch_size: int, max_seq_len: int = 0,
                     prefill_op_data: Optional[Dict[Union[int, Tuple], Tuple[ModelOpField]]] = None,
                     decode_op_data: Optional[Dict[Union[int, Tuple], Tuple[ModelOpField]]] = None) -> Optional[Tuple[
        ModelOpField]]:
        if prefill_op_data is None and decode_op_data is None:
            return None
        # 获取指定batch size的op 信息
        if batch_stage == "prefill":
            op_info = prefill_op_data
        else:
            op_info = decode_op_data
        if len(list(op_info.keys())[0]) == 2:
            op_type = MAX_SEQ_LEN
        else:
            op_type = BATCH_SIZE
        if op_type == MAX_SEQ_LEN:
            _relation_key = (batch_size, max_seq_len)
            if _relation_key in op_info:
                return tuple(op_info[_relation_key])
            else:
                tmp_list = list(op_info.keys())
                if not tmp_list:
                    return None 
                try:
                    _cur_batch = min(range(len(tmp_list)), key=lambda i: abs(tmp_list[i][0] - batch_size))
                    tmp_list_filtered = [k for k in tmp_list if k[0] == tmp_list[_cur_batch][0]]
                    if not tmp_list_filtered:
                        return None  
                    _cur_max_seq_len = min(range(len(tmp_list_filtered)), \
                                           key=lambda i: abs(tmp_list_filtered[i][1] - max_seq_len))
                    return tuple(op_info[tmp_list_filtered[_cur_max_seq_len]])
                except IndexError as e:
                    logger.error(f"error occurred when get_op_filed for max_seq_len: {e}")
                    return None  
        else:
            if (batch_size,) in op_info:
                return tuple(op_info[(batch_size,)])
            else:
                tmp_list = list(op_info.keys())
                if not tmp_list:
                    
                    return None  
                try:
                    _cur_index = min(range(len(tmp_list)), key=lambda i: abs(tmp_list[i][0] - batch_size))
                    return tuple(op_info[tmp_list[_cur_index]])
                except IndexError:
                    # 捕捉索引越界错误
                    logger.error(f"error occurred when get_op_filed for max_batch_size: {e}")
                    return None  

    def load_static_data(self):
        if self.data_file_paths.hardware_path.exists():
            self.hardware = self.load_hardware_data(self.data_file_paths.hardware_path)
        if self.data_file_paths.env_path.exists():
            self.env_info = self.load_env_data(self.data_file_paths.env_path)
        if self.data_file_paths.mindie_config_path.exists():
            self.mindie_info = self.load_mindie_config(self.data_file_paths.mindie_config_path)
        if self.data_file_paths.config_path.exists():
            self.model_config_info = self.load_model_config(self.data_file_paths.config_path)
        if self.data_file_paths.model_struct_path.exists():
            self.model_struct_info = self.load_model_struct(self.data_file_paths.model_struct_path)
        if self.data_file_paths.model_prefill_op_path.exists():
            self.prefill_op_data = self.load_op_data(self.data_file_paths.model_prefill_op_path)
        if self.data_file_paths.model_decode_op_path.exists():
            self.decode_op_data = self.load_op_data(self.data_file_paths.model_decode_op_path)
