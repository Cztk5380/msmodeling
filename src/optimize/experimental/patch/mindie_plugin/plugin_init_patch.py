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

PLUGIN_WHITE_LIST = [*PLUGIN_WHITE_LIST, 'simulate']


class SimulatePluginParameterValidator(PluginParameterValidator):
    def __init__(self, speculation_gamma):
        super().__init__(speculation_gamma)
        self.rules['simulate'] = {
            PLUGIN_FIELDS: set(),  # 没有额外的字段要求
            PLUGIN_CHECK_FUNC: lambda data: True  # 不需要额外校验
        }


PluginParameterValidator = SimulatePluginParameterValidator
