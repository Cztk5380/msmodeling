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

from loguru import logger

from msserviceprofiler.modelevalstate.common import get_module_version

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
    from msserviceprofiler.modelevalstate.patch.patch_manager import Patch2rc1

    simulate_patch.append(Patch2rc1)
    optimize_patch.append(Patch2rc1)
except ImportError as e:
    logger.warning(f"Failed from .patch_manager import Patch2rc1. error: {e}")

try:
    from msserviceprofiler.modelevalstate.patch.patch_vllm import PatchVllm

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
