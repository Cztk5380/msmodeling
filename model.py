# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import dataclasses
from typing import Optional, Protocol

import stime
from device import DeviceConfig


class ModelConfig:
    """
    Model configuration describes the model architecture, parallel strategy, etc.
    """

    def __init__(
        self, model_name: Optional[str] = None, num_dp_partitions: int = 1, **kwargs
    ):
        self.model_name = model_name
        self.num_dp_partitions = num_dp_partitions
        self.kwargs = kwargs


@dataclasses.dataclass
class ModelInput:
    pass


@dataclasses.dataclass
class ModelOutput:
    pass


class Model(Protocol):
    def forward(self, model_input: ModelInput) -> ModelOutput:
        ...
    

class DummyModel(Model):
    def __init__(self, **kwargs):
        self.duration = kwargs.get("duration", 0.01)

    def forward(self, model_input: ModelInput) -> ModelOutput:
        with stime.Duration(self.duration):
            return ModelOutput()


class ModelBuilder:
    @staticmethod
    def build(device: DeviceConfig, dp_rank: int, config: ModelConfig) -> Model:
        if config.model_name is None:
            return DummyModel(**config.kwargs)
        raise NotImplementedError(f"Model {config.model_name} is not implemented.")
