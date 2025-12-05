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

from pathlib import Path
 
from loguru import logger
from msserviceprofiler.msguard import Rule 
from msserviceprofiler.modelevalstate.patch.patch_manager import check_flag, add_patch

 
_patch_dir = Path(__file__).absolute().expanduser().parent.resolve()
 
 
class PatchVllm:
 
    @staticmethod
    def check_version(target_version):
        return True
 
    @staticmethod
    def patch():
        import vllm_ascend
        file_path = vllm_ascend.__path__[0]
        # 检查文件是否存在
        model_runner_file = Path(file_path).joinpath("worker/model_runner.py").resolve()
        if not Rule.input_file_read.is_satisfied_by(model_runner_file):
            logger.error("not found patch file for mindie")
            return
        plugin_manager_patch = _patch_dir.joinpath("model_runner_patch.patch")
        diff_flag = check_flag(model_runner_file, plugin_manager_patch)
        if not diff_flag:
            # 已经打过补丁，不需要打了
            logger.info("The patch already exists.")
            return
        add_patch(model_runner_file, plugin_manager_patch)

