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
from abc import ABC, abstractmethod
from experimental.optimizer.interfaces.custom_process import BaseDataField, CustomProcess
from experimental.config.config import PerformanceIndex

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