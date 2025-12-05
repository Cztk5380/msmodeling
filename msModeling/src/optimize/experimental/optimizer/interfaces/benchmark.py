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
from abc import ABC, abstractmethod
from msserviceprofiler.modelevalstate.optimizer.interfaces.custom_process import BaseDataField, CustomProcess
from msserviceprofiler.modelevalstate.config.config import PerformanceIndex

MS_TO_S = 10 ** 3
US_TO_S = 10 ** 6


class BenchmarkInterface(CustomProcess, BaseDataField, ABC):
    """
    操作benchmark程序，测试性能。
    """
    @property
    def num_prompts(self) -> int:
        """
        获取服务的进程名属性
        Returns:""

        """
        return 0

    @num_prompts.setter
    def num_prompts(self, value):
        """
        获取服务的进程名属性
        Returns:""

        """
        pass

    @property
    def model_name(self) -> str:
        """
        获取当前运行运行模型的名字
        Returns:

        """
        return ""

    @property
    def dataset_path(self) -> str:
        """
        获取当前使用的数据集
        Returns:

        """
        return ""

    @property
    def max_output_len(self) -> 0:
        """
        获取当前设置的最大输出长度。
        Returns:

        """
        return 0

    @abstractmethod
    def update_command(self) -> None:
        """
        更新服务启动命令。更新self.command属性。
        Returns: None

        """
        pass

    @abstractmethod
    def get_performance_index(self) -> PerformanceIndex:
        """
        获取性能指标
        Returns: 指标数据类

        """
        pass