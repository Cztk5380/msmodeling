# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from dataclasses import dataclass
from typing import List, Optional

import yaml


# ------------------ dataclass for instance-level configuration ------------------
# TOBEDONE: fit tensorcast module
@dataclass
class ParallelConfig:
    tp_size: int = 1
    num_dp_partitions: int = 1


@dataclass
class InstanceConfig:
    num_instances: int
    num_devices_per_instance: int
    pd_role: str  # "prefill" / "decode" / "both"
    parallel_config: ParallelConfig


# ------------------ dataclass for common configuration ------------------
# TOBEDONE: fit tensorcast module
@dataclass
class ModelConfig:
    name: str
    head_dim: int
    num_heads: int
    precision_bytes: int
    num_layers: int
    unit_time: Optional[float] = None
    duration: Optional[float] = None


@dataclass
class LoadGenConfig:
    load_gen_type: str
    num_requests: int
    num_input_tokens: int
    num_output_tokens: int
    request_rate: float


@dataclass
class ServingConfig:
    max_concurrency: int = 100


@dataclass
class CommonConfig:
    model_config: ModelConfig
    load_gen: LoadGenConfig
    serving_config: ServingConfig


class Config:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, parsed_args):
        if not self._initialized:
            self.instance_config_list = self._parse_instance_config(
                parsed_args.instance_config_path
            )
            # 2. common 也改为 YAML
            self.common_config = self._parse_common_config(
                parsed_args.common_config_path
            )
            self._initialized = True

    @staticmethod
    def _parse_common_config(path: str) -> CommonConfig:
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        model = ModelConfig(**d.pop("model_config", {}))
        load_gen = LoadGenConfig(**d.pop("load_gen", {}))
        serving = ServingConfig(**d.pop("serving_config", {}))
        return CommonConfig(model_config=model, load_gen=load_gen, serving_config=serving)

    @staticmethod
    def _parse_instance_config(path: str) -> List[InstanceConfig]:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        instances = raw.get("instance_groups", [])
        return [
            InstanceConfig(
                parallel_config=ParallelConfig(**item.pop("parallel_config", {})),
                **item
            )
            for item in instances
        ]

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            raise ValueError("config not initialized")
        return cls._instance