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
from typing import Type

from loguru import logger

from msserviceprofiler.modelevalstate.optimizer.interfaces.benchmark import BenchmarkInterface
from msserviceprofiler.modelevalstate.optimizer.interfaces.simulator import SimulatorInterface

simulates = {}
benchmarks = {}


def register_simulator(model_arch: str,
                       model_cls: Type[SimulatorInterface],
                       ) -> None:
    """
    Register an external model to be used in modelevalstate.

    :code:`model_cls` can be either:

    - A :class:`SimulatorInterface` class directly referencing the model.
    """
    if not isinstance(model_arch, str):
        msg = f"`model_arch` should be a string, not a {type(model_arch)}"
        raise TypeError(msg)

    if model_arch in simulates:
        logger.warning(
            f"Model architecture {model_arch} is already registered, and will be "
            "overwritten by the new model class {model_cls}.")
    if isinstance(model_cls, type) and issubclass(model_cls, SimulatorInterface):
        simulates[model_arch] = model_cls
    else:
        msg = ("`model_cls` should be a SimulatorInterface class, "
               f"not a {type(model_arch)}")
        raise TypeError(msg)


def register_benchmarks(model_arch: str,
                        model_cls: Type[BenchmarkInterface],
                        ) -> None:
    """
    Register an external model to be used in modelevalstate.

    :code:`model_cls` can be either:

    - A :class:`BenchmarkInterface` class directly referencing the model.
    """
    if not isinstance(model_arch, str):
        msg = f"`model_arch` should be a string, not a {type(model_arch)}"
        raise TypeError(msg)

    if model_arch in benchmarks:
        logger.warning(
            f"Model architecture {model_arch} is already registered, and will be "
            "overwritten by the new model class {model_cls}.")
    if isinstance(model_cls, type) and issubclass(model_cls, BenchmarkInterface):
        benchmarks[model_arch] = model_cls
    else:
        msg = ("`model_cls` should be a BenchmarkInterface class, "
               f"not a {type(model_arch)}")
        raise TypeError(msg)


def register_ori_functions():
    from msserviceprofiler.modelevalstate.optimizer.plugins.benchmark import VllmBenchMark, AisBench
    from msserviceprofiler.modelevalstate.optimizer.plugins.simulate import VllmSimulator, Simulator

    register_benchmarks("vllm_benchmark", VllmBenchMark)
    register_benchmarks("ais_bench", AisBench)
    register_simulator("vllm", VllmSimulator)
    register_simulator("mindie", Simulator)
