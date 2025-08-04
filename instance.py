# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
import itertools
from typing import Dict, List
from collections import deque
import logging

from device import Device, MachineConfig, MachineManager
from engine import EngineLoadBalancer, Engine
from model import ModelConfig
from request import Request, RequestState
import stime

logger = stime.get_logger(__name__)


class Instance(ABC):
    id_counter = itertools.count()

    def __init__(self, machine_config: MachineConfig, model_config: ModelConfig, pd_role: str):
        self.id = next(self.id_counter)
        self.machine_manager = MachineManager(machine_config)
        self.machine_config = machine_config
        self.model_config = model_config
        if pd_role not in ['prefill', 'decode', 'both']:
            raise ValueError("In instance __init__, pd_role should be one of ['prefill', 'decode', 'both'] " \
                "but got pd_role: %s", pd_role)
        self.pd_role = pd_role
        self.requests: Dict[int, Request] = {}
        if self.machine_config.num_devices % self.model_config.num_dp_partitions != 0:
            raise ValueError("In instance __init__, num_devices must be divisible by num_dp_partitions," \
                "but got num_devices = %d, num_dp_partitions = %d", \
                    self.machine_config.num_devices, self.model_config.num_dp_partitions)
        self.num_devices_per_dp = self.machine_config.num_devices // self.model_config.num_dp_partitions
        self.engines: List[Engine] = [
            Engine(self.machine_manager.get_devices()[i * self.num_devices_per_dp:(i + 1) * self.num_devices_per_dp], 
                   dp_rank=i, model_config=model_config, pd_role=pd_role)
            for i in range(model_config.num_dp_partitions)
        ]
        self.load_balacer = EngineLoadBalancer(self.engines)

        # serving metrics
        self.max_concurrent_requests = 0

    def handle(self, request: Request):
        logger.debug("Instance %d capacity %d handling %s", self.id, len(self.requests), request)
        if request.id in self.requests:
            raise ValueError("In Instance handle, request.id already in self.requests")

        if self.pd_role == 'prefill':
            if request.state != RequestState.ARRIVES_SERVER:
                raise ValueError("In Instance handle, pd_role is prefill. " \
                    "request.state should be ARRIVES_SERVER, but get %s", request.state)
            request.state = RequestState.PREFILLING
            request.prefill_done_signal.connect(self._on_prefill_done)
        elif self.pd_role == 'both':
            if request.state != RequestState.ARRIVES_SERVER:
                raise ValueError("In Instance handle, pd_role is both. " \
                    "request.state should be ARRIVES_SERVER, but get %s", request.state)
            request.state = RequestState.PREFILLING
            request.decode_done_signal.connect(self._on_decode_done)
        elif self.pd_role == 'decode':
            if request.state != RequestState.PREFILL_DONE:
                raise ValueError("In Instance handle, pd_role is decode. " \
                    "request.state should be PREFILL_DONE, but get %s", request.state)
            request.state = RequestState.DECODING
            request.decode_done_signal.connect(self._on_decode_done)
        else:
            raise ValueError

        self.requests[request.id] = request
        self.max_concurrent_requests = max(self.max_concurrent_requests, len(self.requests))
        engine = self.load_balacer.select(request)
        engine.handle(request)

    def get_work_load(self):
        if self.pd_role == 'prefill':
            return sum(request.num_input_tokens for request in self.requests.values())
        elif self.pd_role == 'decode':
            return len(self.requests)
        elif self.pd_role == 'both':
            return sum(request.num_input_tokens if request.state == RequestState.PREFILLING else 1 \
                for request in self.requests.values())
        else:
            raise ValueError("In Instance get_work_load, self.pd_role should be one of ['prefill', 'decode', 'both'] " \
                "but got self.pd_role: %s", self.pd_role)

    def _on_prefill_done(self, request: Request):
        if request.id not in self.requests:
            raise ValueError("In Instance _on_prefill_done, request.id not in self.requests")
        self.requests.pop(request.id)

    def _on_decode_done(self, request: Request):
        if request.id not in self.requests:
            raise ValueError("In Instance _on_decode_done, request.id not in self.requests")
        self.requests.pop(request.id)


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
