# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import List
import threading

from device import Device
from model import ModelConfig, ModelInput, ModelOutput, Model, ModelBuilder
from request import Request
import stime


class Worker:
    def __init__(self, device: Device, dp_rank: int, model_config: ModelConfig):
        """The worker instantiaties a model to compute on a device"""
        # TOBEDONE: build model according to model configuration
        self.model: Model = ModelBuilder.build(device, dp_rank, model_config) # TOBEDONE: check device is right input

    @staticmethod
    def _preprocess_input(self, batch: List[Request]) -> ModelInput:
        # TOBEDONE
        return ModelInput()

    def run(self, batch: List[Request]):
        model_input: ModelInput = self._preprocess_input(batch)
        model_output: ModelOutput = self.model.forward(model_input)
        # mark EOS etc.
        self._postprocess_output(model_output, batch)
    
    def _postprocess_output(self, output: ModelOutput, batch: List[Request]):
        # TOBEDONE
        pass

class ModelRunner:
    def __init__(self, devices: List[Device], dp_rank: int, model_config: ModelConfig):
        """
        Each model runner works on a data-parallel partition ('dp_rank') given a list of devices
        to compute the model given the model configuration 'model_config'. The computation is
        further sharded among the 'devices' via other parallel algorithms like tensor parallel etc.
        It instantiates a list of 'Workers' with each working on a device.
        """
        self.workers: List[Worker] = [Worker(device, dp_rank, model_config) for device in devices]

    def process_batch(self, batch: List[Request]):
        def worker_run(worker: Worker):
            # TOBEDONE: consider input copy overhead
            worker.run(batch)

        worker_threads = [stime.Thread(target=worker_run, args=(worker,)) for worker in self.workers]
        for worker_thread in worker_threads:
            worker_thread.start()
        for worker_thread in worker_threads:
            worker_thread.join()
            


