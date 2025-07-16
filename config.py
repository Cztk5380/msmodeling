# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from enum import Enum, auto


class PdDeploymentPolicy(Enum):
    DISAGGREGRATE = auto()

PD_DEPLOYMENT_POLICY = PdDeploymentPolicy.DISAGGREGRATE
NUM_PREFILL_INSTANCES = 2
NUM_DECODE_INSTANCES = 4
