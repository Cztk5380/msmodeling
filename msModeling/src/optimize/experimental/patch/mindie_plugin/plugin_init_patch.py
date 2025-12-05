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

PLUGIN_WHITE_LIST = [*PLUGIN_WHITE_LIST, 'simulate']


class SimulatePluginParameterValidator(PluginParameterValidator):
    def __init__(self, speculation_gamma):
        super().__init__(speculation_gamma)
        self.rules['simulate'] = {
            PLUGIN_FIELDS: set(),  # 没有额外的字段要求
            PLUGIN_CHECK_FUNC: lambda data: True  # 不需要额外校验
        }


PluginParameterValidator = SimulatePluginParameterValidator
