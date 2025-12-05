# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import bisect
import json
import os
import time
from collections.abc import Callable
from copy import deepcopy
from enum import Enum
from inspect import isfunction
from math import isinf, isclose
from pathlib import Path
from typing import Any, List, Tuple, Type, Optional, Union

import numpy as np
from loguru import logger
from pydantic import BaseModel, field_validator, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, TomlConfigSettingsSource

import msserviceprofiler.modelevalstate
from msserviceprofiler.modelevalstate.common import is_vllm, is_mindie, ais_bench_exists
from msserviceprofiler.modelevalstate.config.custom_command import BenchmarkCommandConfig, VllmBenchmarkCommandConfig, \
    MindieCommandConfig, VllmCommandConfig, AisBenchCommandConfig, KubectlCommandConfig
from msserviceprofiler.msguard.security import open_s, mkdir_s
from .base_config import (
    INSTALL_PATH, RUN_PATH, ServiceType, CUSTOM_OUTPUT, DeployPolicy, RUN_TIME,	
    modelevalstate_config_path, MODEL_EVAL_STATE_CONFIG_PATH, AnalyzeTool, BenchMarkPolicy,
)


class MetricAlgorithm(BaseModel):
    metric: str = "TTFT"
    algorithm: str = "average"


class PerformanceConfig(BaseModel):
    time_to_first_token: MetricAlgorithm = MetricAlgorithm(metric="TTFT",
                                                           algorithm="average")
    time_per_output_token: MetricAlgorithm = MetricAlgorithm(metric="TPOT",
                                                             algorithm="average")


dtype_func = {"int": int, "float": float}


class Stage(Enum):
    start = "start"
    running = "running"
    error = "error"
    stop = "stop"


class ProcessState(BaseModel):
    stage: Stage = Stage.start
    info: str = ""


class OptimizerConfigField(BaseModel):
    name: str = "max_batch_size"
    config_position: str = "BackendConfig.ScheduleConfig.maxBatchSize"
    min: float = 0.0
    max: float = 100.0
    dtype: str = "float"
    value: Union[int, float, bool] = 0.0
    dtype_param: Any = None
    constant: Optional[float] = None  # 识别是否是常量

    @model_validator(mode="after")
    def update_constant(self):
        if self.min > self.max:
            raise ValueError(f"min({self.min}) > max({self.max}). please check")
        # 如果min 等于max 但是 constant 没有设置，自动设置constant 为最大值。
        if self.constant and not isclose(self.min, self.max):
            self.min = self.max = self.constant
        elif self.constant is None and isclose(self.min, self.max, rel_tol=1e-5) and self.dtype in dtype_func.keys():
            self.constant = dtype_func.get(self.dtype, float)(self.max)
 
        return self
 
    def convert_dtype(self, value):
        return dtype_func.get(self.dtype, float)(value)
 
    def find_available_value(self, value):
        _new_value = dtype_func.get(self.dtype, float)(value)
        if self.dtype == "enum":
            if value in self.dtype_param:
                return value
            else:
                _index = bisect.bisect_left(self.dtype_param, value)
                if _index == len(self.dtype_param):
                    _new_value = self.dtype_param[-1]
                else:
                    _new_value = self.dtype_param[_index]
                return _new_value
        else:
            if self.min <= _new_value <= self.max:
                return _new_value
            elif _new_value < self.min:
                return dtype_func.get(self.dtype, float)(self.min)
            else:
                return dtype_func.get(self.dtype, float)(self.max)



default_support_field = [
    # max batch size 最小值要大于max_prefill_batch_size的最大值。
    OptimizerConfigField(name="max_batch_size", config_position="BackendConfig.ScheduleConfig.maxBatchSize", min=10,
                         max=1000, dtype="int"),
    OptimizerConfigField(name="max_prefill_batch_size",
                         config_position="BackendConfig.ScheduleConfig.maxPrefillBatchSize", min=0.1, max=0.7,
                         dtype="ratio", dtype_param="max_batch_size"),
    OptimizerConfigField(name="prefill_time_ms_per_req",
                         config_position="BackendConfig.ScheduleConfig.prefillTimeMsPerReq", max=1000, dtype="int"),
    OptimizerConfigField(name="decode_time_ms_per_req",
                         config_position="BackendConfig.ScheduleConfig.decodeTimeMsPerReq", max=1000, dtype="int"),
    OptimizerConfigField(name="support_select_batch",
                         config_position="BackendConfig.ScheduleConfig.supportSelectBatch", max=1,
                         dtype="bool"),
    OptimizerConfigField(name="max_prefill_token",
                         config_position="BackendConfig.ScheduleConfig.maxPrefillTokens", min=4096, max=409600,
                         dtype="int"),
    OptimizerConfigField(name="max_queue_deloy_mircroseconds",
                         config_position="BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds", min=500, max=1000000,
                         dtype="int"),
    OptimizerConfigField(name="prefill_policy_type",
                         config_position="BackendConfig.ScheduleConfig.prefillPolicyType", min=0, max=1,
                         dtype="enum", dtype_param=[0, 1, 3]),
    OptimizerConfigField(name="decode_policy_type",
                         config_position="BackendConfig.ScheduleConfig.decodePolicyType", min=0, max=1,
                         dtype="enum", dtype_param=[0, 1, 3]),
    OptimizerConfigField(name="max_preempt_count",
                         config_position="BackendConfig.ScheduleConfig.maxPreemptCount", min=0, max=1,
                         dtype="ratio", dtype_param="max_batch_size"),
    OptimizerConfigField(name="tp",
                         config_position="BackendConfig.ModelDeployConfig.ModelConfig.0.tp", min=0, max=1,
                         dtype="enum", dtype_param=[1, 2, 4, 8, 16]),
    OptimizerConfigField(name="dp",
                         config_position="BackendConfig.ModelDeployConfig.ModelConfig.0.dp", min=0, max=0,
                         dtype="factories", dtype_param={
            "target_name": "tp",
            "product": 16,
            "dtype": "int"
        }),
    OptimizerConfigField(name="moe_ep",
                         config_position="BackendConfig.ModelDeployConfig.ModelConfig.0.moe_ep", min=0, max=1,
                         dtype="enum", dtype_param=[1, 2, 4, 8, 16]),
    OptimizerConfigField(name="moe_tp",
                         config_position="BackendConfig.ModelDeployConfig.ModelConfig.0.moe_tp", min=0, max=0,
                         dtype="factories", dtype_param={
            "target_name": "moe_ep",
            "product": 16,
            "dtype": "int"
        }),
]


def range_to_enum(params_field: Tuple[OptimizerConfigField, ...]):
    for v in params_field:
        if v.dtype != "range":
            continue
        if not v.dtype_param:
            continue
        try:
            _start = int(v.min)
            _end = int(v.max)
            _step = int(v.dtype_param)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed convert to int data, data: {v.min, v.max, v.dtype_param}")
            continue
        _enums = [j for j in range(_start, _end + _step, _step)]
        v.min = 0
        v.max = 1
        v.dtype_param = _enums
        v.dtype = "enum"


def update_optimizer_value(params_field: Tuple[OptimizerConfigField, ...],
                           simulate_run_info: Tuple[OptimizerConfigField, ...], support_select_is_false):
    for i, v in enumerate(params_field):
        if v.dtype == "ratio":
            _field = simulate_run_info[i]
            _t_op = [_op for _op in simulate_run_info if _op.name == v.dtype_param][0]
            _field.value = int(_field.value * _t_op.value)
        if v.dtype == "factories":
            _field = simulate_run_info[i]
            _t_op = [_op for _op in simulate_run_info if _op.name == v.dtype_param["target_name"]][0]
            if _t_op.value != 0:
                _field.value = dtype_func.get(v.dtype_param["dtype"], int)(v.dtype_param["product"] / _t_op.value)
        if "maxPrefillBatchSize" in v.config_position:
            _field = simulate_run_info[i]
            if _field.value == 0:
                _field.value = 1
        if support_select_is_false:
            # prefillTimeMsPerReq和decodeTimeMsPerReq在“supportSelectBatch”设置为“true”时生效。
            _field = simulate_run_info[i]
            if "prefillTimeMsPerReq" in _field.config_position:
                _field.value = 0
            if "decodeTimeMsPerReq" in _field.config_position:
                _field.value = 0


def map_param_with_value(params: np.ndarray, params_field: Tuple[OptimizerConfigField, ...]):
    _simulate_run_info = []
    _support_select_is_false = False
    i = 0
    for v in params_field:
        _field = deepcopy(v)
        if _field.constant is not None or isclose(_field.min, _field.max, rel_tol=1e-5):
            if _field.value and not isinf(_field.value):
                try:
                    _field.value = dtype_func.get(v.dtype, int)(_field.value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed in func {params[i]} for {v}, error: {e}")
            _simulate_run_info.append(_field)
            continue
        if v.dtype == "int":
            try:
                _field.value = int(params[i])
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed convert to int data, data: {params[i]}")
                _field.value = params[i]
        elif v.dtype == "bool":
            if params[i] > 0.5:
                _field.value = True
                if "supportSelectBatch" in _field.name:
                    _support_select_is_false = True
            else:
                _field.value = False
        elif v.dtype == "enum":
            segment = np.linspace(v.min, v.max, len(v.dtype_param) + 1)
            if params[i] <= v.min:
                _field.value = v.dtype_param[0]
            elif params[i] >= v.max:
                _field.value = v.dtype_param[-1]
            else:
                _enum_index = np.searchsorted(segment, params[i]) - 1
                _field.value = v.dtype_param[_enum_index]
        elif v.dtype == "share":
            for _op in _simulate_run_info:
                if _op.name == v.dtype_param:
                    _field.value = int(_op.min + _op.max - _op.value)
        else:
            try:
                _field.value = float(params[i])
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed convert to float data, data: {params[i]}")
                _field.value = params[i]
        i += 1
        _simulate_run_info.append(_field)
    update_optimizer_value(params_field, tuple(_simulate_run_info), _support_select_is_false)
    return _simulate_run_info


def reverse_special_field(params_field: Tuple[OptimizerConfigField, ...], params: np.ndarray,
                          concurrency: int):
    _params = params
    i = 0
    for v in params_field:
        # 常量设置值了 或者最大值和最小值一样 说明这个参数为常量。
        if v.constant is not None or isclose(v.min, v.max, rel_tol=1e-5):
            continue
        if v.dtype == "ratio":
            for _op in params_field:
                if _op.name == v.dtype_param and _op.value != 0:
                    _t_op = _op
                    _params[i] = float(v.value / _t_op.value)
        if v.name in ["CONCURRENCY", "MAXCONCURRENCY"]:
            if v.value == 0 and v.dtype == "ratio":
                # CONCURRENCY 字段 是某个对象的百分比时，并且 值为0，说明第一次，设置为0
                _params[i] = 1
            elif v.value is not None and v.dtype == "ratio" and concurrency > 0:
                _params[i] = v.value / concurrency
            elif v.value is not None:
                # 原来的方式 int
                _params[i] = v.value
            else:
                # 不是百分比时，
                _params[i] = concurrency
        i += 1
    return _params


def field_to_param(params_field: Tuple[OptimizerConfigField, ...]):
    concurrency = request_rate = None
    _params = []
    for _, v in enumerate(params_field):
        if v.constant is not None or isclose(v.min, v.max, rel_tol=1e-5):
            continue
        if v.dtype == "int":
            try:
                _params.append(int(v.value))
            except Exception as e:
                logger.warning(f"Failed in field to param, error: {e}")
                _params.append(v.value)
        elif v.dtype == "bool":
            if v.value:
                _params.append(1)
            else:
                _params.append(0)
        elif v.dtype == "enum":
            # 不存在的值 将其放进去，再进行转换。
            if v.value not in v.dtype_param and isinstance(v.value, str):
                v.dtype_param.append(v.value)
            if v.value not in v.dtype_param and isinstance(v.value, (int, float)):
                v.dtype_param.sort()
                bisect.insort_left(v.dtype_param, v.value)
            _index = v.dtype_param.index(v.value)
            segment = np.linspace(v.min, v.max, len(v.dtype_param) + 1)
            _params.append((segment[_index] + segment[_index + 1]) / 2)
        else:
            _params.append(v.value)
        if v.config_position == "BackendConfig.ScheduleConfig.maxBatchSize" or v.name in ["MAX_NUM_SEQS",
                                                                                          "max_batch_size"]:
            concurrency = v.value
    _params = np.array(_params, dtype=float)
    return reverse_special_field(params_field, _params, concurrency)


class PerformanceIndex(BaseModel):
    generate_speed: Optional[float] = None
    time_to_first_token: Optional[float] = None
    time_per_output_token: Optional[float] = None
    success_rate: Optional[float] = None
    throughput: Optional[float] = None


class BenchMarkConfig(BaseModel):
    process_name: str = "benchmark"
    output_path: Path = Path("benchmark")
    work_path: Path = Field(default_factory=lambda: Path(os.getcwd()).resolve())
    command: BenchmarkCommandConfig = BenchmarkCommandConfig()
    performance_config: PerformanceConfig = PerformanceConfig()
    target_field: List[OptimizerConfigField] = Field(default_factory=list)

    @field_validator("work_path")
    @classmethod
    def check_dir(cls, path: Path) -> Path:
        if not path.exists():
            logger.error(f"FileNotFound: {path!r}")
        return path


class ProfileConfig(BaseModel):
    output: Path = Path("benchmark")
    profile_input_path: Path = Field(
        default_factory=lambda data: data["output"].joinpath("profile_input_path").resolve(),
        validate_default=True
    )
    profile_output_path: Path = Field(
        default_factory=lambda data: data["output"].joinpath("profile_output_path").resolve(),
        validate_default=True
    )

    @field_validator("profile_input_path", "profile_output_path")
    @classmethod
    def create_path(cls, path: Path) -> Path:
        mkdir_s(path)
        return path


class CommunicationConfig(BaseModel):
    base_path: Path = Path("communication")
    cmd_file: Optional[Path] = Field(
        default_factory=lambda data: data["base_path"].joinpath("cmd.txt").resolve())
    res_file: Optional[Path] = Field(
        default_factory=lambda data: data["base_path"].joinpath("res.txt").resolve())


class DataStorageConfig(BaseModel):
    store_dir: Path = Path("store")
    pso_top_k: int = 3

    @field_validator("store_dir")
    @classmethod
    def create_path(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True, mode=0o750)
        return path


class LatencyModel(BaseModel):
    base_path: Path = Path("latency_model")
    model_path: Optional[Path] = Field(
        default_factory=lambda data: data["base_path"].joinpath("bak/base/xgb_model.ubj").resolve())
    static_file_dir: Optional[Path] = Field(
        default_factory=lambda data: data["base_path"].joinpath("model_static_file").resolve(), validate_default=True)
    req_and_decode_file: Optional[Path] = Field(
        default_factory=lambda data: data["base_path"].joinpath("req_id_and_decode_num.json").resolve())
    cache_data: Optional[Path] = Field(
        default_factory=lambda data: data["base_path"].joinpath("cache").resolve())

    @field_validator("base_path", "cache_data", "static_file_dir")
    @classmethod
    def create_path(cls, path: Path) -> Path:
        mkdir_s(path)
        return path


class MindieConfig(BaseModel):
    # 运行mindie时，要修改的mindie config
    process_name: str = "mindie, mindie-llm, mindieservice_daemon, mindie_llm"
    output: Path = Path("mindie")
    work_path: Path = Field(default_factory=lambda: Path(os.getcwd()).resolve())
    config_path: Path = Path("/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json")
    config_bak_path: Path = Path("/usr/local/Ascend/mindie/latest/mindie-service/conf/config_bak.json")
    command: MindieCommandConfig = MindieCommandConfig()
    target_field: List[OptimizerConfigField] = default_support_field


class KubectlConfig(BaseModel):
    process_name: str = ""
    kubectl_default_path: Path = Path("")
    kubectl_single_path: Optional[Path] = Field(
        default_factory=lambda data: data["kubectl_default_path"].joinpath("deploy.sh").resolve())
    config_single_path: Optional[Path] = Field(
        default_factory=lambda data: data["kubectl_default_path"].joinpath("conf/config.json").resolve())
    config_single_pd_path: Optional[Path] = Field(
        default_factory=lambda data: data["kubectl_default_path"].joinpath("conf/ms_controller.json").resolve())
    config_single_pd_bak_path: Optional[Path] = Field(
        default_factory=lambda data: data["kubectl_default_path"].joinpath("conf/ms_controller_bak.json").resolve())
    config_single_bak_path: Optional[Path] = Field(
        default_factory=lambda data: data["kubectl_default_path"].joinpath("conf/config_bak.json").resolve())
    delete_path: Optional[Path] = Field(
        default_factory=lambda data: data["kubectl_default_path"].joinpath("delete.sh").resolve())
    work_path: Path = Field(default_factory=lambda: Path(os.getcwd()).resolve())
    command: KubectlCommandConfig = Field(
        default_factory=lambda data: KubectlCommandConfig(kubectl_default_path=data["kubectl_default_path"]))
    target_field: List[OptimizerConfigField] = Field(default_factory=list)


class AisBenchConfig(BaseModel):
    process_name: str = "ais_bench"
    output_path: Path = Path("ais_bench")
    work_path: Path = Field(default_factory=lambda: Path(os.getcwd()).resolve())
    command: AisBenchCommandConfig = AisBenchCommandConfig()
    performance_config: PerformanceConfig = PerformanceConfig()
    target_field: List[OptimizerConfigField] = Field(default_factory=list)
    model: str = ""
    path: str = ""
    host_ip: str = ""
    host_port: int = 0
    max_out_len: int = 0
    best_concurrency_coefficient: int = 3
    best_concurrency_threshold: int = 200


class VllmBenchmarkConfig(BaseModel):
    output_path: Path = Path("vllm")
    process_name: str = ""
    command: VllmBenchmarkCommandConfig = VllmBenchmarkCommandConfig()
    performance_config: PerformanceConfig = PerformanceConfig()
    target_field: List[OptimizerConfigField] = Field(default_factory=list)


class VllmConfig(BaseModel):
    output: Path = Path("vllm")
    process_name: str = "vllm"
    work_path: Path = Field(default_factory=lambda: Path(os.getcwd()).resolve())
    command: VllmCommandConfig = VllmCommandConfig()
    target_field: List[OptimizerConfigField] = Field(default_factory=list)


class PsoOptions(BaseModel):
    c1: float = 2.0  # 推荐范围 0-4, c1 c2 2, c1 1.6和c2 1.8, c1 1.6 和c2 2
    c2: float = 2.0
    w: float = 1.8  # 推荐范围0.4,2， 典型取值，0.9  1.2 1.5  1.8


class PsoStrategy(BaseModel):
    # 支持 exp_decay, nonlin_mod, lin_variation, random
    w: str = "exp_decay"
    c1: str = "exp_decay"
    c2: str = "exp_decay"


class Settings(BaseSettings):
    """
    设置类的定义，通过读取配置文件初始化配置
    """
    model_config = SettingsConfigDict(
        toml_file=[INSTALL_PATH.joinpath("model_eval_state.toml"), Path("~/model_eval_state.toml").expanduser(),
                   RUN_PATH.joinpath("model_eval_state.toml"),
                   INSTALL_PATH.joinpath("config.toml"), Path("~/config.toml").expanduser(),
                   RUN_PATH.joinpath("config.toml"), modelevalstate_config_path],
        env_prefix="model_eval_state_")

    output: Path = Field(default_factory=lambda: Path(os.getcwd()).joinpath("result").resolve(), validate_default=True)
    simulator_output: Path = Field(
        default_factory=lambda data: data["output"].joinpath("simulator").resolve())
    pso_options: PsoOptions = PsoOptions()
    pso_strategy: PsoStrategy = PsoStrategy()
    particles_time_out: int = 1 * 60 * 60
    n_particles: int = Field(default=5, gt=0, lt=1000)
    iters: int = Field(default=10, gt=0, lt=1000)
    ftol: float = -np.inf
    ftol_iter: int = 1
    ttft_penalty: float = 3.0  # 惩罚系数
    tpot_penalty: float = 3.0
    success_rate_penalty: float = 5.0
    ttft_slo: float = Field(default=0.5, gt=0)
    tpot_slo: float = Field(default=0.05, gt=0)
    success_rate_slo: float = Field(default=1.0, gt=0)
    slo_coefficient: float = 0.1
    generate_speed_target: float = 5000.0
    sample_size: Optional[int] = None
    mem_coefficient: float = 0.8
    max_fine_tune: int = 10
    scaling_coefficient: float = 1.3
    step_size: float = 0.6
    theory_guided_enable: bool = True
    service: str = ServiceType.master.value
    communication: CommunicationConfig = Field(
        default_factory=lambda data: CommunicationConfig(base_path=data["output"].joinpath("communication")),
        validate_default=True)
    kubectl: KubectlConfig = KubectlConfig()
    latency_model: LatencyModel = Field(
        default_factory=lambda data: LatencyModel(base_path=data["output"].joinpath("latency_model")),
        validate_default=True)
    vllm: VllmConfig = Field(default_factory=lambda data: VllmConfig(output=data["output"].joinpath("vllm")),
                             validate_default=True)
    mindie: MindieConfig = Field(default_factory=lambda data: MindieConfig(output=data["output"].joinpath("mindie")),
                                 validate_default=True)
    kubectl: KubectlConfig = Field(default_factory=lambda data: KubectlConfig(output=data["output"].joinpath("k8s")),
                                 validate_default=True)
    ais_bench: AisBenchConfig = AisBenchConfig()
    benchmark: BenchMarkConfig = BenchMarkConfig()

    vllm_benchmark: VllmBenchmarkConfig = VllmBenchmarkConfig()
    profile: ProfileConfig = ProfileConfig()


    data_storage: DataStorageConfig = Field(
        default_factory=lambda data: DataStorageConfig(store_dir=data["output"].joinpath("store")),
        validate_default=True)

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: Type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, TomlConfigSettingsSource(settings_cls), file_secret_settings)

    @field_validator("output", "simulator_output")
    @classmethod
    def create_path(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True, mode=0o750)
        return path

    @model_validator(mode="after")
    def partial_update_vllm(self):
        if not is_vllm():
            return self
        output = VllmConfig.model_fields["output"].default
        if self.vllm.output == output:
            self.vllm.output = self.output.joinpath(output)
        output = VllmBenchmarkConfig.model_fields["output_path"].default
        if self.vllm_benchmark.output_path == output:
            self.vllm_benchmark.output_path = self.output.joinpath(output)
        if self.vllm_benchmark.command.result_dir == VllmBenchmarkCommandConfig.model_fields["result_dir"].default:
            self.vllm_benchmark.command.result_dir = str(self.vllm_benchmark.output_path.joinpath("result"))
        Path(self.vllm_benchmark.command.result_dir).mkdir(parents=True, exist_ok=True, mode=0o750)

        self.vllm_benchmark.command.host = self.vllm.command.host
        self.vllm_benchmark.command.port = self.vllm.command.port
        self.vllm_benchmark.command.model = self.vllm.command.model
        self.vllm_benchmark.command.served_model_name = self.vllm.command.served_model_name
        if self.vllm.target_field:
            range_to_enum(self.vllm.target_field)
        if self.vllm_benchmark.target_field:
            range_to_enum(self.vllm_benchmark.target_field)
        return self

    @model_validator(mode="after")
    def partial_update_aisbench(self):
        if not ais_bench_exists():
            return self
        output = AisBenchConfig.model_fields["output_path"].default
        if self.ais_bench.output_path == output:
            self.ais_bench.output_path = self.output.joinpath(output)
        if not self.ais_bench.command.work_dir:
            self.ais_bench.command.work_dir = str(self.ais_bench.output_path)
        if self.ais_bench.target_field:
            range_to_enum(self.ais_bench)
        return self

    @model_validator(mode="after")
    def partial_update_mindie(self):
        range_to_enum(self.mindie.target_field)
        if not is_mindie():
            return self
        if not self.mindie.config_path.exists():
            logger.error(f"File Not Found. file: {self.mindie.config_path!r}")
            return self
        with open_s(self.mindie.config_path, "r") as f:
            try:
                mindie_config = json.load(f)
            except json.decoder.JSONDecodeError as e:
                logger.error(f"Failed in load {self.mindie.config_path!r}. error: {e}")
                raise e
        # 从mindie config 中获取可能获取的端口信息，模型信息。
        ip_address = mindie_config["ServerConfig"]["ipAddress"]
        port = mindie_config["ServerConfig"]["port"]
        management_ip_address = mindie_config["ServerConfig"]["managementIpAddress"]
        management_port = mindie_config["ServerConfig"]["managementPort"]
        model_name = mindie_config["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["modelName"]
        model_path = mindie_config["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["modelWeightPath"]
        output = MindieConfig.model_fields["output"].default
        if self.mindie.output == output:
            self.mindie.output = self.output.joinpath(output)
        output_path = BenchMarkConfig.model_fields["output_path"].default
        if self.benchmark.output_path == output_path:
            self.benchmark.output_path = self.output.joinpath(output_path)
        if not self.benchmark.command.http:
            self.benchmark.command.http = f"http://{ip_address}:{port}"
        if not self.benchmark.command.management_http:
            self.benchmark.command.management_http = f"http://{management_ip_address}:{management_port}"
        if not self.benchmark.command.model_name:
            self.benchmark.command.model_name = model_name
        if not self.benchmark.command.model_path:
            self.benchmark.command.model_path = model_path
        if not self.benchmark.command.save_path:
            self.benchmark.command.save_path = str(self.benchmark.output_path.joinpath("instance"))
        mkdir_s(Path(self.benchmark.command.save_path)) 
        if self.profile.output == ProfileConfig.model_fields["output"].default:
            self.profile = ProfileConfig(output=self.output.joinpath("profile"))
        return self


custom_settings_func: Optional[Callable] = None

settings = None


def get_settings() -> Settings:
    """
    获取 settings 对象
    Return: Settings()实例
    """
    global settings
    if not settings:
        if custom_settings_func and isfunction(custom_settings_func):
            settings = custom_settings_func()
        else:
            settings = Settings()
    return settings


def register_settings(func: Optional[Callable] = None) -> None:
    """
    注册自定义settings  可提供函数生成或提供新的settings
    """
    global custom_settings_func
    custom_settings_func = func