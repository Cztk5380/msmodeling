# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from enum import Enum, auto


class PdDeploymentPolicy(Enum):
    DISAGGREGRATE = auto()

pd_deployment_policy = PdDeploymentPolicy.DISAGGREGRATE
num_prefill_instances = 2
num_decode_instances = 4
