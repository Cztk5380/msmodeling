# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import List, Optional

import stime
from service_sim.device import Device
from service_sim.request import Request

logger = stime.get_logger(__name__)


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


class ModelRunner:
    def __init__(self, devices: List[Device], dp_rank: int, model_config: ModelConfig):
        """
        Each model runner works on a data-parallel partition ('dp_rank') given a list of devices
        to compute the model given the model configuration 'model_config'. The computation is
        further sharded among the 'devices' via other parallel algorithms like tensor parallel etc.
        It instantiates a list of 'Workers' with each working on a device.
        """
        self.devices = devices
        self.dp_rank = dp_rank
        self.model_config = model_config

    def process_batch(self, batch: List[Request]):
        duration = self._get_estimate_time(batch)
        with stime.Duration(duration):
            logger.debug(f"{self.model_config.model_name} process batch, batch length: {len(batch)}")

    def _get_estimate_time(self, batch: List[Request]):
        if self.model_config.kwargs.get("duration") is not None:
            return self.model_config.kwargs.get("duration")
        else:
            raise NotImplementedError("get_estimate_time func is not implemented")
