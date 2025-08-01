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

    def __init__(self, machine_config: MachineConfig, model_config: ModelConfig):
        self.id = next(self.id_counter)
        self.machine_manager = MachineManager(machine_config)
        self.machine_config = machine_config
        self.model_config = model_config
        self.requests: Dict[int, Request] = {}
        if self.machine_config.num_devices % self.model_config.num_dp_partitions != 0:
            raise ValueError("In instance __init__, num_devices must be divisible by num_dp_partitions," \
                f"but got num_devices = %d, num_dp_partitions = %d", \
                    self.machine_config.num_devices, self.model_config.num_dp_partitions)
        self.num_devices_per_dp = self.machine_config.num_devices // self.model_config.num_dp_partitions
        self.engines = []

        # servingh metrics
        self.max_concurrent_requests = 0

    def get_work_load(self):
        return sum(engine.get_work_load() for engine in self.engines)

    @abstractmethod
    def handle(self, request: Request):
        return
    

class PrefillInstance(Instance):
    id_counter = itertools.count()

    def __init__(self, machine_config: MachineConfig, model_config: ModelConfig):
        super().__init__(machine_config, model_config)
        self.engines: List[Engine] = [
            Engine(self.machine_manager.get_devices()[i * self.num_devices_per_dp:(i + 1) * self.num_devices_per_dp], 
                   dp_rank=i, model_config=model_config, pd_role="prefill")
            for i in range(model_config.num_dp_partitions)
        ]
        self.load_balacer = EngineLoadBalancer(self.engines)

    def handle(self, request: Request):
        logger.debug("Prefill instance %d capacity %d handling %s", self.id, len(self.requests), request)
        if request.id in self.requests:
            raise ValueError("In PrefillInstance handle, request.id already in self.requests")
        if request.state != RequestState.ARRIVES_SERVER:
            raise ValueError("In PrefillInstance handle, request.state should be ARRIVES_SERVER," \
                "but get %s", request.state)
        request.state = RequestState.PREFILLING
        self.requests[request.id] = request
        self.max_concurrent_requests = max(self.max_concurrent_requests, len(self.requests))
        request.prefill_done_signal.connect(self._on_prefill_done)
        engine = self.load_balacer.select(request)
        engine.handle(request)

    def _on_prefill_done(self, request: Request):
        if request.id not in self.requests:
            raise ValueError("In PrefillInstance _on_prefill_done, request.id not in self.requests")
        self.requests.pop(request.id)

    
class DecodeInstance(Instance):
    id_counter = itertools.count()

    def __init__(self, machine_config: MachineConfig, model_config: ModelConfig):
        super().__init__(machine_config, model_config)
        self.engines: List[Engine] = [
            Engine(self.machine_manager.get_devices()[i * self.num_devices_per_dp:(i + 1) * self.num_devices_per_dp], 
                   dp_rank=i, model_config=model_config, pd_role="decode")
            for i in range(model_config.num_dp_partitions)
        ]
        self.load_balancer = EngineLoadBalancer(self.engines)
        
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

class PrefillDecodeInstance(Instance):
    id_counter = itertools.count()

    def __init__(self, machine_config: MachineConfig, model_config: ModelConfig):
        super().__init__(machine_config, model_config)
        self.engines: List[Engine] = [
            Engine(self.machine_manager.get_devices()[i * self.num_devices_per_dp:(i + 1) * self.num_devices_per_dp], 
                   dp_rank=i, model_config=model_config, pd_role="both")
            for i in range(model_config.num_dp_partitions)
        ]
        self.load_balancer = EngineLoadBalancer(self.engines)
        
    def handle(self, request: Request):
        logger.debug("PrefillDecode instance %d capacity %d handling %s", self.id, len(self.requests), request)
        if request.id in self.requests:
            raise ValueError("request.id in self.requests")
        if request.state != RequestState.ARRIVES_SERVER:
            raise ValueError("request.state != RequestState.ARRIVES_SERVER")
        request.state = RequestState.PREFILLING
        self.requests[request.id] = request
        self.max_concurrent_requests = max(self.max_concurrent_requests, len(self.requests))
        request.decode_done_signal.connect(self._on_decode_done)
        engine = self.load_balacer.select(request)
        engine.handle(request)

    def _on_decode_done(self, request: Request):
        if request.id not in self.requests:
            raise ValueError("request.id not in self.requests")
        self.requests.pop(request.id)


class InstanceLoadBalancer:
    def __init__(self, instances: List[Instance]):
        self.instances = instances

    def select(self, request: Request) -> Instance:
        # greedily choose the instance having the least total number of input tokens to handle
        # TOBEDONE: support  heterogeneous instances
        work_loads = [instance.get_work_load() for instance in self.instances]
        # logger.debug("PrefillInstanceLoadBalancer.select: %d", work_loads)
        min_value = min(work_loads)
        min_index = work_loads.index(min_value)
        return self.instances[min_index]
