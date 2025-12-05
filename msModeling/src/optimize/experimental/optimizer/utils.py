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
import os
import shutil
from pathlib import Path

import psutil
from loguru import logger

from msserviceprofiler.msguard import Rule
from msserviceprofiler.msguard.security import walk_s


def remove_file(output_path: Path):
    if not output_path:
        return
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    if not output_path.exists():
        return
    if output_path.is_file():
        output_path.unlink()
        return
    for file in output_path.iterdir():
        if file.is_file():
            file.unlink()
        else:
            try:
                shutil.rmtree(file)
            except OSError:
                logger.error(f"remove file failed, file_path :{output_path!r}")
                return


def kill_children(children):
    for child in children:
        if not child.is_running():
            continue
        try:
            child.send_signal(9)
            child.wait(10)
        except Exception as e:
            logger.error(f"Failed to kill the {child.pid} process. detail: {e}")
            continue

        if child.is_running():
            logger.error(f"Failed to kill the {child.pid} process.")


def kill_process(process_name):
    for proc in psutil.process_iter(["pid", "name"]):
        if not hasattr(proc, "info"):
            continue
        if process_name not in proc.info["name"]:
            continue
        children = psutil.Process(proc.pid).children(recursive=True)
        kill_children([proc])
        kill_children(children)


def backup(target, bak, class_name="", max_depth=10, current_depth=0):
    if not target or not bak:
        return
    if not isinstance(target, Path):
        target = Path(target)
    if not isinstance(bak, Path):
        bak = Path(bak)
    if not target.exists() or not bak.exists():
        return
    if current_depth >= max_depth:
        logger.warning(f"Reached maximum backup depth {max_depth} for {target}")
        return

    new_file = bak.joinpath(class_name).joinpath(target.name)
    if target.is_file():
        if not Rule.input_file_read.is_satisfied_by(target):
            return
        new_file.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        if not new_file.exists():
            shutil.copy(target, new_file)
    else:
        if not Rule.input_dir_traverse.is_satisfied_by(target):
            return
        if new_file.exists():
            for child in new_file.iterdir():
                backup(child, new_file, class_name, max_depth, current_depth + 1)
        else:
            shutil.copytree(target, new_file)


def close_file_fp(file_fp):
    if not file_fp:
        return
    try:
        # 检查file_fp是否是一个文件对象
        if hasattr(file_fp, 'close'):
            file_fp.close()
        else:
            # 如果file_fp是一个文件描述符，调用os.close()
            os.close(file_fp)
    except (AttributeError, OSError):
        return
    

def get_folder_size(folder_path: Path) -> int:
    folder = Path(folder_path)
    if not folder.exists():
        return 0
    total_size = 0
    for file_path in walk_s(folder):
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)

    return total_size




def get_required_field_from_json(data, key, max_depth=20, current_depth=0):
    """
    data: json 形式的多层嵌套对象
    key: 要获取的字段名，多层之间用.号连接，
    """
    if current_depth > max_depth:
        raise ValueError(f"Recursive depth exceeded maximum allowed depth of {max_depth}")
    _cur_key = key
    _next_key = None
    if "." in key:
        _index = key.find(".")
        _cur_key = key[:_index]
        _next_key = key[_index + 1:]
    _value = None
    if isinstance(data, dict):
        _value = data[_cur_key]
    elif isinstance(data, list):
        _value = data[int(_cur_key)]
    else:
        raise ValueError(f"Unsupported data type: {data}, please confirm. ")
    if _next_key:
        return get_required_field_from_json(_value, _next_key, max_depth, current_depth + 1)
    else:
        return _value
    

def is_root():
    return os.name != 'nt' and os.getuid() == 0