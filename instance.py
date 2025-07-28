# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
import itertools
from typing import Dict, List
from collections import deque
import logging

from device import Device, MachineConfig, MachineManager
from engine import DecodeEngineLoadBalancer, PrefillEngineLoadBalacer, Engine
from model import ModelConfig
from request import Request, RequestState
import stime

logger = stime.getLogger(__name__)


class Instance(ABC):
    id_counter = itertools.count()

    def __init__(self, machine_config: MachineConfig, model_config: ModelConfig):
        self.id = next(self.id_counter)
        self.machine_manager = MachineManager(machine_config)
        self.machine_config = machine_config
        self.model_config = model_config
        self.requests: Dict[int, Request] = {}
        if self.machine_config.num_devices % self.model_config.num_dp_partitions != 0:
            raise ValueError("self.machine_config.num_devices mod self.model_config.num_dp_partitions != 0")
        num_devices_per_dp = self.machine_config.num_devices // self.model_config.num_dp_partitions
        self.engines: List[Engine] = [
            Engine(self.machine_manager.get_devices()[i * num_devices_per_dp:(i + 1) * num_devices_per_dp], 
                   dp_rank=i, model_config=model_config)
            for i in range(model_config.num_dp_partitions)
        ]

        # servingh metrics
        self.max_concurrent_requests = 0

    @abstractmethod
    def handle(self, request: Request):
        return
    

class PrefillInstance(Instance):
    id_counter = itertools.count()

    def __init__(self, machine_config: MachineConfig, model_config: ModelConfig):
        super().__init__(machine_config, model_config)
        self.load_balacer = PrefillEngineLoadBalacer(self.engines)

    def handle(self, request: Request):
        logger.debug("Prefill instance %d capacity %d handling %s", self.id, len(self.requests), request)
        if request.id in self.requests:
            raise ValueError("request.id in self.requests")
        if request.state != RequestState.ARRIVES_SERVER:
            raise ValueError("request.state != RequestState.ARRIVES_SERVER")
        request.state = RequestState.PREFILLING
        self.requests[request.id] = request
        self.max_concurrent_requests = max(self.max_concurrent_requests, len(self.requests))
        request.prefill_done_signal.connect(self._on_prefill_done)
        engine = self.load_balacer.select(request)
        engine.handle(request)

    def _on_prefill_done(self, request: Request):
        if request.id not in self.requests:
            raise ValueError("request.id not in self.requests")
        self.requests.pop(request.id)

    
class DecodeInstance(Instance):
    id_counter = itertools.count()

    def __init__(self, machine_config: MachineConfig, model_config: ModelConfig):
        super().__init__(machine_config, model_config)
        self.load_balancer = DecodeEngineLoadBalancer(self.engines)
        
    def handle(self, request: Request):
        logger.debug("Decode instance %d capacity %d handling %s", self.id, len(self.requests), request)
        if request.id in self.requests:
            raise ValueError("request.id in self.requests")
        if request.state != RequestState.PREFILL_DONE:
            raise ValueError("request.state != RequestState.PREFILL_DONE")
        request.state = RequestState.DECODING
        self.requests[request.id] = request
        self.max_concurrent_requests = max(self.max_concurrent_requests, len(self.requests))
        request.decode_done_signal.connect(self._on_decode_done)
        engine = self.load_balancer.select(request)
        engine.handle(request)

    def _on_decode_done(self, request: Request):
        if request.id not in self.requests:
            raise ValueError("request.id not in self.requests")
        self.requests.pop(request.id)


class PrefillInstaceLoadBalacer:
    def __init__(self, instances: List[Instance]):
        self.instances = instances

    def select(self, request: Request) -> Instance:
        # greedily choose the instance having the least total number of input tokens to handle
        # TOBEDONE: support  heterogeneous instances
        sums = []
        for instance in self.instances:
            requests = list(instance.requests.values())
            sums.append(sum(request.num_input_tokens for request in requests))
        logger.debug("PrefillInstanceLoadBalancer.select: %d", sums)
        min_value = min(sums)
        min_index = sums.index(min_value)
        return self.instances[min_index]
    

class DecodeInstanceLoadBalancer:
    def __init__(self, instances: List[Instance]):
        self.instances = instances
        
    def select(self, request: Request) -> Instance:
        # greedily choose the instance having the least total of requests to handle
        # TOBEDONE: support heterogeneous instances
        sums = [len(instance.requests) for instance in self.instances]
        logger.debug("DecodeInstanceLoadBalancer.select: %d", sums)
        min_value = min(sums)
        min_index = sums.index(min_value)
        return self.instances[min_index]
