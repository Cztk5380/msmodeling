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
"""
 通过设置环境变量 来决定是如何对代码打补丁。

# 直接修改原代码
1. 进行仿真 MODEL_EVAL_STATE_SIMULATE
> export MODEL_EVAL_STATE_SIMULATE=True

2. 进行寻优 MODEL_EVAL_STATE_ALL
> export MODEL_EVAL_STATE_ALL=True

"""
import os
import traceback

try:
    from loguru import logger
except ModuleNotFoundError as e:
    from logging import getLogger

    logger = getLogger(__name__)

MODEL_EVAL_STATE_SIMULATE = "MODEL_EVAL_STATE_SIMULATE"
MODEL_EVAL_STATE_ALL = "MODEL_EVAL_STATE_ALL"


def dispatch(target_env):
    target_flag = os.getenv(target_env) or os.getenv(target_env.lower())
    if target_flag and (target_flag.lower() == "true" or target_flag.lower() != "false"):
        from experimental.patch import enable_patch
        if enable_patch(target_env):
            logger.info("The collected patch is successfully installed.")
    else:
        logger.debug(f"{target_env}: {target_flag}")


try:
    dispatch(MODEL_EVAL_STATE_SIMULATE)
    dispatch(MODEL_EVAL_STATE_ALL)
except Exception as e:
    logger.error(f"Failed in Import patch. {e}")
    traceback.print_exc()