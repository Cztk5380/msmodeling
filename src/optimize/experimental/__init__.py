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
import sys

from loguru import logger

# 增加 MODELEVALSTATE_LEVEL设置日志级别，ERROR， INFO DEBUG
log_level = os.getenv("MODELEVALSTATE_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=log_level, enqueue=True)