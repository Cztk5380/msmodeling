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
from contextlib import contextmanager
from typing import Tuple, Optional

import requests

from msserviceprofiler.modelevalstate.config.config import OptimizerConfigField, ProcessState, Stage
from msserviceprofiler.modelevalstate.optimizer.interfaces.custom_process import BaseDataField, CustomProcess


class SimulatorInterface(CustomProcess, BaseDataField, ABC):
    """
    操作服务框架。用于操作服务相关功能。
    """

    @property
    @abstractmethod
    def base_url(self) -> str:
        """
        获取服务的base url 属性
        Returns:

        """
        pass

    @abstractmethod
    def update_command(self) -> None:
        """
        更新服务启动命令。更新self.command属性。
        Returns: None

        """
        pass

    def update_config(self, params: Optional[Tuple[OptimizerConfigField]] = None) -> bool:
        """
        根据参数更新服务的配置文件，或者其他配置，服务启动前根据传递的参数值 修改配置文件。使得新的配置生效。
        Args:
            params: 调优参数列表，是一个元祖，根据其中每一个元素的value和config position进行定义。

        Returns: None

        """
        return True

    def stop(self, del_log: bool = True):
        """
        运行时，其他的准备工作。
        Returns:

        """
        pass


    def health(self) -> ProcessState:
        """
        获取当前服务的状态.
        当前实现为基础的vllm url
        Returns: None

        """
        process_res = super().health()
        if process_res.stage == Stage.error:
            return process_res
        try:
            res = requests.get(self.base_url, timeout=10)
        except requests.exceptions.RequestException as e:
            return ProcessState(stage=Stage.error, info=str(e))
        else:
            if res.status_code == 200:
                return ProcessState(stage=Stage.running)
            else:
                return ProcessState(stage=Stage.error, info=f"return code {res.status_code}. text {res.text}")

    @contextmanager
    def enable_simulation_model(self):
        """
        启动使用仿真模拟的模型进行推理来代替真实模型进行推理。
        Returns: None

        """
        # 开启仿真模型代替真实模型
        yield True
        # 关闭仿真模型代替真实模型
