# instance.py
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import itertools
from abc import ABC
from typing import Dict, List

import stime
from device import MachineConfig, MachineManager
from engine import Engine, EngineLoadBalancer
from model import ModelConfig
from request import Request, RequestState


logger = stime.get_logger(__name__)


class Instance(ABC):
    id_counter = itertools.count()

    def __init__(self, machine_config: MachineConfig, model_config: ModelConfig):
        self.id = next(self.id_counter)
        self.machine_manager = MachineManager(machine_config)
        self.machine_config = machine_config
        self.model_config = model_config
        if self.machine_config.num_devices % self.model_config.num_dp_partitions != 0:
            raise ValueError(
                "In instance __init__, num_devices must be divisible by num_dp_partitions,"
                "but got num_devices = %d, num_dp_partitions = %d"
                % (self.machine_config.num_devices, self.model_config.num_dp_partitions)
            )
        num_devices_per_dp = (
            self.machine_config.num_devices // self.model_config.num_dp_partitions
        )
        self.engines: List[Engine] = [
            Engine(
                self.machine_manager.get_devices()[
                    i * num_devices_per_dp: (i + 1) * num_devices_per_dp
                ],
                dp_rank=i,
                model_config=model_config,
            )
            for i in range(model_config.num_dp_partitions)
        ]
        self.load_balancer = EngineLoadBalancer(self.engines)


    def handle(self, request: Request):
        logger.debug("Instance %d handling %s", self.id, request)
        if request.state not in [RequestState.ARRIVES_SERVER, RequestState.KVS_TRANSFERRING]:
            raise ValueError("Instance.handle failed, request.state should be ARRIVES_SERVER " \
                "or KVS_TRANSFERRING, but get %s" % request.state)

        if request.state == RequestState.ARRIVES_SERVER:
            request.state = RequestState.PREFILLING
        else:
            request.state = RequestState.DECODING

        engine = self.load_balancer.select(request)
        engine.handle(request)

    def get_work_load(self):
        return sum(engine.get_work_load() for engine in self.engines)


class InstanceLoadBalancer:
    def __init__(self, instances: List[Instance]):
        self.instances = instances

    def select(self, request: Request) -> Instance:
        # greedily choose the instance having the least total number of input tokens to handle
        # TOBEDONE: support  heterogeneous instances
        work_loads = [instance.get_work_load() for instance in self.instances]
        min_value = min(work_loads)
        min_index = work_loads.index(min_value)
        return self.instances[min_index]
