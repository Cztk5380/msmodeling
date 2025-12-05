# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
import stat
from pathlib import Path

from loguru import logger
from packaging import version
from msserviceprofiler.msguard import validate_params, Rule
from msserviceprofiler.msguard.security import open_s

_patch_dir = Path(__file__).absolute().expanduser().parent.resolve()


def check_flag(target_file, patch_file):
    with open_s(target_file, "r", encoding="utf-8") as f:
        data = f.readlines()
    with open_s(patch_file, "r", encoding="utf-8") as f:
        patch_data = f.readlines()
    i = 0
    diff_flag = True
    for _o_row in data:
        # 原来的代码
        if i == 0 and _o_row != patch_data[0]:
            continue
        if i >= len(patch_data):
            break
        # 发现有补丁代码
        if _o_row == patch_data[i]:
            i += 1
            diff_flag = False
        else:
            diff_flag = True
    return diff_flag
 

@validate_params({'target_file': Rule.output_path_write, 'patch_file': Rule.input_file_read})
def add_patch(target_file, patch_file):
    with open_s(patch_file, "r", encoding="utf-8") as f:
        patch_data = f.readlines()
    try:
        with open_s(target_file, "a", encoding="utf-8") as f:
            f.writelines(patch_data)
            f.flush()
    except Exception as e:
        logger.error(f"add patch failed, error: {e}")
        raise e

    # 没有打补丁的，添加补丁文件内容
    logger.info("The patch is installed successfully.")


class Patch2rc1:
    mindie_llm = "2.2"
    mindie_llm_low = "2.1rc1"

    @staticmethod
    def check_version(target_version):
        _t_v = version.parse(target_version)
        _c_v_up = version.parse(Patch2rc1.mindie_llm)
        _c_v_low = version.parse(Patch2rc1.mindie_llm_low)
        if _c_v_low < _t_v <= _c_v_up:
            pass
        else:
            logger.warning("The version may not match.")
        return True

    @staticmethod
    def patch():
        import mindie_llm
        file_path = mindie_llm.__path__[0]
        # 检查文件是否存在
        simulate_plugin_file = Path(file_path).joinpath("text_generator/plugins/simulate/simulate_plugin.py").resolve()
        simulate_init_file = Path(file_path).joinpath("text_generator/plugins/simulate/__init__.py").resolve()
        plugins_init_file = Path(file_path).joinpath("text_generator/plugins/__init__.py").resolve()
        if not Rule.input_file_read.is_satisfied_by(plugins_init_file):
            raise FileNotFoundError(plugins_init_file)
        if not simulate_init_file.parent.exists():
            simulate_init_file.parent.mkdir(parents=True, mode=0o750)

        simulate_plugin_patch = _patch_dir.joinpath("mindie_plugin/simulate/simulate_plugin.py")
        simulate_init_patch = _patch_dir.joinpath("mindie_plugin/simulate/__init__.py")
        if not simulate_init_file.exists():
            shutil.copy(simulate_init_patch, simulate_init_file)
        if not simulate_plugin_file.exists():
            shutil.copy(simulate_plugin_patch, simulate_plugin_file)
        plugins_init_patch = _patch_dir.joinpath("mindie_plugin/plugin_init_patch.py")
        diff_flag = check_flag(plugins_init_file, plugins_init_patch)
        if not diff_flag:
            # 已经打过补丁，不需要打了
            logger.info("The patch aleady exists.")
            return
        add_patch(plugins_init_file, plugins_init_patch)

