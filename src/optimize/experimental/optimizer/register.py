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
from typing import Type

from loguru import logger

from experimental.optimizer.interfaces.benchmark import BenchmarkInterface
from experimental.optimizer.interfaces.simulator import SimulatorInterface

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
    from experimental.optimizer.plugins.benchmark import VllmBenchMark, AisBench
    from experimental.optimizer.plugins.simulate import VllmSimulator, Simulator

    register_benchmarks("vllm_benchmark", VllmBenchMark)
    register_benchmarks("ais_bench", AisBench)
    register_simulator("vllm", VllmSimulator)
    register_simulator("mindie", Simulator)
