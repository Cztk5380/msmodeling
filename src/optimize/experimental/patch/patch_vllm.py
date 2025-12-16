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

from pathlib import Path
 
from loguru import logger
from msguard import Rule 
from experimental.patch.patch_manager import check_flag, add_patch

 
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

