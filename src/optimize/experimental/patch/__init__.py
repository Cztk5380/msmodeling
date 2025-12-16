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

from loguru import logger

from experimental.common import get_module_version

MINDIE_LLM = "mindie_llm"
VLLM_ASCEND = "vllm_ascend"

simulate_patch = []
optimize_patch = []
vllm_simulate_patch = []
vllm_optimize_patch = []

env_patch = {
    "MODEL_EVAL_STATE_SIMULATE": simulate_patch,
    "MODEL_EVAL_STATE_ALL": optimize_patch
}

vllm_env_patch = {
    "MODEL_EVAL_STATE_SIMULATE": vllm_simulate_patch,
    "MODEL_EVAL_STATE_ALL": vllm_optimize_patch
}

try:
    from experimental.patch.patch_manager import Patch2rc1

    simulate_patch.append(Patch2rc1)
    optimize_patch.append(Patch2rc1)
except ImportError as e:
    logger.warning(f"Failed from .patch_manager import Patch2rc1. error: {e}")

try:
    from experimental.patch.patch_vllm import PatchVllm

    vllm_optimize_patch.append(PatchVllm)
    vllm_simulate_patch.append(PatchVllm)
except ImportError as e:
    logger.warning(f"Failed from .patch_vllm import PatchVllm. error: {e}")


def enable_patch(target_env):
    flag = []
    try:
        mindie_llm_version = get_module_version(MINDIE_LLM)

        for _p in env_patch.get(target_env, []):
            if _p.check_version(mindie_llm_version):
                _p.patch()
                flag.append(_p)
    except (ModuleNotFoundError, ValueError):
        pass

    try:
        vllm_ascend_version = get_module_version(VLLM_ASCEND)
        for _p in vllm_env_patch.get(target_env):
            if _p.check_version(vllm_ascend_version):
                _p.patch()
                flag.append(_p)
    except (ModuleNotFoundError, ValueError):
        pass

    if flag:
        logger.info(f"Installed patch list {flag}.")
