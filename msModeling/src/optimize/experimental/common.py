# !/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple
import shutil
import pandas as pd
from loguru import logger
from msserviceprofiler.msguard import validate_params, Rule


_PREFILL = "prefill"
_DECODE = "decode"
_KEY_WORD = "H" + "B" + "M"


class StateType(Enum):
    DEFAULT = 0
    LINE = 1


@dataclass
class State:
    prefill: int = 0
    decode: int = 0
    batch_prefill: int = 0
    batch_decode: int = 0

    def __repr__(self):
        return f"TT_{self.prefill}_{self.decode}_{self.batch_prefill}_{self.batch_decode}"

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __ne__(self, other):
        return self.__repr__() != other.__repr__()

    def sum(self):
        return self.prefill + self.decode + self.batch_prefill + self.batch_decode


def computer_speed(line_node, field):
    if getattr(line_node, field) == 0:
        return 1
    return 1 / (getattr(line_node, field) * 10 ** -3)


def computer_speed_with_second(line_node, field):
    if getattr(line_node, field) == 0:
        return 1
    return 1 / (getattr(line_node, field) * 10 ** -6)


def get_train_sub_path(base_path: Path):
    # 给训练输出目录生成新的目录
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True, mode=0o750)
    _sub_len = len([0 for _ in base_path.iterdir()])
    _sub_dir = base_path.joinpath(f"{_sub_len + 1}")
    _sub_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
    return _sub_dir


def update_global_coefficient(global_coefficient: Dict, key: State, value: float) -> None:
    if key not in global_coefficient:
        global_coefficient[key] = [value]
    else:
        global_coefficient[key].append(value)


def get_module_version(module_name):
    try:
        # 方法1：直接导入模块
        import importlib
        module = importlib.import_module(module_name)
        if hasattr(module, "__version__"):
            return module.__version__
        elif hasattr(module, "version"):
            return module.version
    except ImportError:
        pass

    try:
        # 方法2：使用 pkg_resources
        import pkg_resources
        return pkg_resources.get_distribution(module_name).version
    except (ImportError, pkg_resources.DistributionNotFound):
        pass

    try:
        # 方法3：使用 importlib.metadata（Python 3.8+）
        import importlib.metadata
        return importlib.metadata.version(module_name)
    except (ImportError):
        pass

    # # 方法4：最后尝试 pip show
    _flag = "MODEL_EVAL_STATE_GET_MODULE_VERSION_FLAG"
    try:
        if os.getenv(_flag) == "true":
            raise ValueError
        pip_path = shutil.which("pip3")
        if pip_path is not None:
            output = subprocess.check_output(
                [pip_path, "show", module_name],
                universal_newlines=True,
                env={"MODEL_EVAL_STATE_GET_MODULE_VERSION_FLAG": "true"}
            )
            for line in output.splitlines():
                if line.startswith("Version:"):
                    return line.split(":")[1].strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise ValueError("模块未安装或无法获取版本")


@validate_params({"path": Rule.input_file_read})
def read_csv_s(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to read csv %r." % path) from e


def is_mindie():
    try:
        import mindie_llm
    except ModuleNotFoundError:
        return False
    return True


def is_vllm():
    try:
        import vllm
    except ModuleNotFoundError:
        return False
    return True


def ais_bench_exists():
    try:
        import ais_bench
    except ModuleNotFoundError:
        return False
    return True


def get_npu_total_memory(device_id: int = 0) -> Tuple[int, int]:
    _npu_smi_path = shutil.which("npu-smi")
    if not _npu_smi_path:
        raise ValueError("Not Found npu-smi command path. ")
    _id_map_cmd = ["npu-smi", "info", "-m"]
    cmd = ["npu-smi", "info", "-t", "usages"]
    try:
        _map_out = subprocess.check_output(_id_map_cmd).decode("utf-8")
        _npu_id = _chip_id = 0
        for _line in _map_out.split("\n"):
            if not _line.strip():
                continue
            _result = _line.split()
            _npu_id, _chip_id, _chip_logic_id, *_chip_other = _result
            if _chip_logic_id.strip() == str(device_id):
                break
        if not _npu_id.isdigit():
            raise ValueError(f"_npu_id {_npu_id} is not a digit.")
        if not _chip_id.isdigit():
            raise ValueError(f"_chip_id {_chip_id} is not a digit.")
        cmd.extend(["-i", _npu_id, "-c", _chip_id])
        output = subprocess.check_output(cmd).decode("utf-8")
        memory_key_word = _KEY_WORD + " Capacity(MB)"	
        usage_rate_key_word = _KEY_WORD + " Usage Rate(%)"
        try:
            total_memory_line = [line for line in output.splitlines() if memory_key_word in line][0]
            memory_usage_rate = [line for line in output.splitlines() if usage_rate_key_word in line][0]
        except IndexError: 
            total_memory_line = [line for line in output.splitlines() if "DDR Capacity(MB)" in line][0]
            memory_usage_rate = [line for line in output.splitlines() if "DDR Hugepages Usage Rate(%)" in line][0]
        total_memory_line = total_memory_line.split(":")[1].strip()
        memory_usage_rate = memory_usage_rate.split(":")[1].strip()
 
        logger.debug(f"cmd: {cmd}, result: {int(total_memory_line), int(memory_usage_rate)}")
        return int(total_memory_line), int(memory_usage_rate)
    except Exception as e:
        logger.error(f"Failed to retrieve total video memory. Please check if the video memory query command {cmd} "
                     f"matches the current parsing code. ")
        raise e
