# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from contextlib import contextmanager
from copy import deepcopy
from math import inf, isinf, isnan, isclose
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger

from msserviceprofiler.modelevalstate.common import is_vllm, is_mindie
from msserviceprofiler.modelevalstate.config.base_config import (
    EnginePolicy, DeployPolicy, AnalyzeTool, REAL_EVALUATION, 
    ServiceType, BenchMarkPolicy, PDPolicy, REQUESTRATES,
    simulate_flag, CONCURRENCYS
)
from msserviceprofiler.modelevalstate.optimizer.register import benchmarks, simulates
from msserviceprofiler.modelevalstate.optimizer.performance_tunner import PerformanceTuner
from msserviceprofiler.modelevalstate.optimizer.utils import get_required_field_from_json, is_root


MAX_ITER_NUM = 200


class PSOOptimizer(PerformanceTuner):
    def __init__(self, scheduler, n_particles: int = 10, iters=100, pso_options=None,
                 target_field: Optional[Tuple] = None, load_history_data: Optional[List] = None,
                 load_breakpoint: bool = False, pso_init_kwargs: Optional[Dict] = None,
                 fine_tune=None, max_fine_tune: int = 10, **kwargs):
        from msserviceprofiler.modelevalstate.config.config import PsoOptions, default_support_field
        super().__init__(**kwargs)
        self.scheduler = scheduler
        self.n_particles = min(n_particles, MAX_ITER_NUM)
        self.iters = min(iters, MAX_ITER_NUM)
        self.target_field = target_field if target_field else default_support_field
        if not pso_options:
            self.pso_options = PsoOptions()
        else:
            self.pso_options = pso_options
        self.load_history_data = load_history_data
        self.load_breakpoint = load_breakpoint
        self.pso_init_kwargs = pso_init_kwargs
        self.init_pos = None
        self.history_cost, self.history_pos = None, None
        self.default_fitness = None
        self.default_run_param = None
        self.default_res = None
        self.sample_data = None
        self.fine_tune = fine_tune
        self.max_fine_tune = min(max_fine_tune, MAX_ITER_NUM)

    @staticmethod
    def is_within_boundary(target_pos, min_bound, max_bound):
        # 检查数据是否再边界内
        for i, v in enumerate(target_pos):
            if min_bound[i] <= v <= max_bound[i]:
                continue
            else:
                return False
        return True

    @staticmethod
    def params_in_records(params, record_params):
        for _his_params in record_params:
            if (_his_params == params).all():
                return True
        return False

    def get_target_field_from_case_data(self, case_data):
        # 有空数据的都视为无效数据跳过。
        _target_field = deepcopy(self.target_field)
        for _field in _target_field:
            _case_value = case_data.get(_field.name, None)
            if _case_value is None:
                raise ValueError("Invalid data.")
            _field.value = _case_value
        return _target_field

    def computer_fitness(self) -> Tuple:
        from msserviceprofiler.modelevalstate.config.config import PerformanceIndex, field_to_param
        all_position = []
        all_cost = []
        _min_bound, _max_bound = self.constructing_bounds()
        for case_data in self.load_history_data:
            _fitness = case_data.get("fitness")
            # 不存在fit ness，则重新计算
            if not _fitness:
                _params = {}
                for k in PerformanceIndex.model_fields.keys():
                    if k in case_data:
                        _params[k] = case_data[k]
                performance_index = PerformanceIndex(**_params)
                _fitness = self.minimum_algorithm(performance_index)
            # 过滤无效的fitness
            if isnan(_fitness) or isinf(_fitness):
                continue
            logger.debug(f"fitness {_fitness}")
            try:
                _target_field = self.get_target_field_from_case_data(case_data)
            except ValueError:
                continue
            _pos = field_to_param(_target_field)
            # 超过最大最小值限制的进行跳过
            if not self.is_within_boundary(_pos, _min_bound, _max_bound):
                continue
            all_cost.append(_fitness)
            all_position.append(_pos)
        if len(all_position) != len(all_cost):
            raise ValueError("Failed in computer_fitness.")
        return all_position, all_cost

    def op_func(self, x) -> np.ndarray:
        n_particles = x.shape[0]
        logger.debug(f"Acquired n_particles: {n_particles}, value: {x}")
        generate_speed = []
        for i in range(n_particles):
            # 调用schedule， 获取采集的数据
            try:
                _res = self.scheduler.run_with_request_rate(x[i], self.target_field)
                # 根据采集的数据，计算最优化值
                _fitness = self.minimum_algorithm(_res)
            except Exception as e:
                logger.error(f"Failed. error: {e}, please check.")
                _fitness = inf
            logger.debug(f"fitness {_fitness}")
            self.scheduler.save_result(fitness=_fitness)
            generate_speed.append(_fitness)
        return np.array(generate_speed)

    def constructing_bounds(self) -> Tuple[Tuple, Tuple]:
        """
        返回示例：((0, 10), (0, 10))
        """
        _min = []
        _max = []
        for _field in self.target_field:
            if _field.constant is not None or isclose(_field.min, _field.max, rel_tol=1e-5):
                continue
            else:
                _min.append(_field.min)
                _max.append(_field.max)
        return (tuple(_min), tuple(_max))

    def dimensions(self):
        d = 0
        for _field in self.target_field:
            if _field.constant is not None or isclose(_field.min, _field.max, rel_tol=1e-5):
                continue
            else:
                d += 1
        return d

    def refine_optimization_candidates(self, best_results: pd.DataFrame):
        from msserviceprofiler.modelevalstate.config.config import field_to_param
        from msserviceprofiler.modelevalstate.optimizer.experience_fine_tunning import StopFineTune
        # 分别精调每组参数
        _record_params = [self.default_run_param]
        _record_res = [self.default_res]
        _record_fitness = [self.default_fitness]
        for _, _pso_info in best_results.iterrows():
            _target_field = self.get_target_field_from_case_data(_pso_info)
            for _field in _target_field:
                if _field.name in REQUESTRATES:
                    _field.value = _field.find_available_value(_field.value * 2)
            params = field_to_param(_target_field)
            # 先全量运行寻优参数
            try:
                _res = self.scheduler.run(params, self.target_field)
                _fitness = self.minimum_algorithm(_res)
            except Exception as e:
                logger.error(f"Runtime exception. error: {e}, please check.")
                _fitness = inf
                self.scheduler.save_result(fitness=_fitness)
                continue
            self.scheduler.save_result(fitness=_fitness)
            _record_params.append(params)
            _record_res.append(_res)
            _record_fitness.append(_fitness)
            # 对寻优参数精调
            self.fine_tune.reset_history()
            for _ in range(self.max_fine_tune):
                try:
                    simulate_run_info = self.fine_tune.fine_tune_with_concurrency_and_request_rate(params, _res)
                except ValueError as e:
                    logger.error("Failed in fine-tuning parameter. error: {e}")
                    break
                except StopFineTune:
                    # 找到满足约束的最好值
                    break
                params = field_to_param(simulate_run_info)
                # 新参数跟之前的参数没变化就提前终止了。
                if self.params_in_records(params, _record_params):
                    break
                try:
                    _res = self.scheduler.run(params, self.target_field)
                    _fitness = self.minimum_algorithm(_res)
                except Exception as e:
                    logger.error(f"Runtime exception. error: {e}, please check.")
                    _fitness = inf
                    self.scheduler.save_result(fitness=_fitness)
                    break
                self.scheduler.save_result(fitness=_fitness)
                _record_params.append(params)
                _record_res.append(_res)
                _record_fitness.append(_fitness)
        return _record_fitness, _record_params, _record_res
    
    def get_max_generate_speed_index(self, performance_index_list, slo_index):
        _best_index = 0
        _max = 0
        for i, v in enumerate(performance_index_list):
            if i not in slo_index:
                continue
            if v.generate_speed > _max:
                _max = v.generate_speed
                _best_index = i
        return _best_index

    def best_params(self, fitnese_list, params_list, performance_index_list):
        from msserviceprofiler.modelevalstate.config.config import get_settings
        # 分析最佳参数
        if not performance_index_list or not fitnese_list or not params_list:
            logger.error(f"Input is empty."
                             f"performance_index_list:{performance_index_list},"
                             f"fitnese_list: {fitnese_list},"
                             f"params_list: {params_list}")
            return None, None, None
        if len(fitnese_list) != len(params_list) != len(performance_index_list):
            logger.error(f"The number of input elements does not match."
                             f"performance_index_list:{len(performance_index_list)},"
                             f"fitnese_list: {len(fitnese_list)},"
                             f"params_list: {len(params_list)}")
            return None, None, None
        for _p in performance_index_list:
            if _p.generate_speed is None:
                _p.generate_speed = 0
            if _p.time_to_first_token is None:
                _p.time_to_first_token = inf
            if _p.time_per_output_token is None:
                _p.time_per_output_token = inf

        if self.tpot_penalty == 0 and self.ttft_penalty == 0:
            _generate_speed = [p.generate_speed for p in performance_index_list]
            _best_index = _generate_speed.index(max(_generate_speed))
            return fitnese_list[_best_index], params_list[_best_index], performance_index_list[_best_index]
        if self.ttft_penalty == 0 and self.tpot_penalty != 0:
            _tpot_threshold = self.fine_tune.tpot_upper_bound
            if _tpot_threshold == 0:
                return fitnese_list[0], params_list[0], performance_index_list[0]
            _tpot_diff = [(p.time_per_output_token - _tpot_threshold) / _tpot_threshold for p in performance_index_list]
            _tpot_lt_slo_index = [i for i, v in enumerate(_tpot_diff) if v < 0]
            # 有满足slo的，从中选吞吐大的
            if _tpot_lt_slo_index:
                _best_index = self.get_max_generate_speed_index(performance_index_list, _tpot_lt_slo_index)
                return fitnese_list[_best_index], params_list[_best_index], performance_index_list[_best_index]
            # 没有满足 slo的，选择离slo最近的约束
            _best_index = _tpot_diff.index(min(_tpot_diff))
            return fitnese_list[_best_index], params_list[_best_index], performance_index_list[_best_index]
        if self.ttft_penalty != 0 and self.tpot_penalty != 0:
            _tpot_threshold = self.fine_tune.tpot_upper_bound
            _ttft_threshold = self.fine_tune.ttft_upper_bound
            if _tpot_threshold == 0 or _ttft_threshold == 0:
                return fitnese_list[0], params_list[0], performance_index_list[0]
            _performance_diff = [((p.time_per_output_token - _tpot_threshold) / _tpot_threshold,
                                  (p.time_to_first_token - _ttft_threshold) / _ttft_threshold) 
                                  for p in performance_index_list]
            # ttft 和 tpot 都满足条件
            _performance_lt_slo_index = [i for i, v in enumerate(_performance_diff) if all([kv < 0 for kv in v])]
            if _performance_lt_slo_index:
                _best_index = self.get_max_generate_speed_index(performance_index_list, _performance_lt_slo_index)
                return fitnese_list[_best_index], params_list[_best_index], performance_index_list[_best_index]
            # 没有满足slo的，选择两个差异和最小的最为最佳值。
            _performance_diff_sum = [sum(v) for v in _performance_diff]
            _best_index = _performance_diff_sum.index(min(_performance_diff_sum))
            return fitnese_list[_best_index], params_list[_best_index], performance_index_list[_best_index]
        # 未知场景，返回第一组作为最佳参数
        return fitnese_list[0], params_list[0], performance_index_list[0]
    
    def mindie_prepare(self, mc):
        from msserviceprofiler.modelevalstate.config.config import get_settings
        settings = get_settings()
        if mc is None:
            return
        if not settings.theory_guided_enable:
            return
        mc.avg_input_length = self.scheduler.benchmark.get_performance_metric("InputTokens")
        mc.max_input_length = self.scheduler.benchmark.get_performance_metric("InputTokens", algorithm="max")
        mc.max_output_length = self.scheduler.benchmark.get_performance_metric("OutputTokens", algorithm="max")
        logger.debug(f"avg_input_length: {mc.avg_input_length}, max_input_length: {mc.max_input_length},"
                     f"max_output_length: {mc.max_output_length}")
        max_batch_size_lb, max_batch_size_ub = mc.get_max_batch_size_bound()
        if not isinf(max_batch_size_ub):
            scale_max_batch_size_ub = int(max_batch_size_ub * settings.scaling_coefficient)
        else:
            scale_max_batch_size_ub = inf
        logger.debug(f"avg_input_length: {mc.avg_input_length}, max_input_length: {mc.max_input_length},"
                     f"max_output_length: {mc.max_output_length}")
        max_batch_size_lb, max_batch_size_ub = mc.get_max_batch_size_bound()
        scale_max_batch_size_ub = int(max_batch_size_ub * settings.scaling_coefficient)
        if max_batch_size_lb >= max_batch_size_ub or max_batch_size_lb <= 0 or max_batch_size_ub <= 0:
            logger.warning(f"Theoretical derivation scope failure.max_batch_size_lb {max_batch_size_lb}, "
                         f"max_batch_size_ub {max_batch_size_ub}, please check env")
            return
        logger.debug(f"max_batch_size_lb {max_batch_size_lb}, max_batch_size_ub {max_batch_size_ub}. "
                     f"scale_max_batch_size_ub {scale_max_batch_size_ub}")

        for _field in self.target_field:
            if _field.name == "max_batch_size":
                # 只有在默认设置的范围内才需要压缩搜索空间
                if _field.min < max_batch_size_lb < _field.max:
                    _field.min = max_batch_size_lb
                if _field.min < scale_max_batch_size_ub < _field.max:
                    _field.max = scale_max_batch_size_ub
                    break
                if _field.min < max_batch_size_ub < _field.max:
                    _field.max = max_batch_size_ub
                break
        logger.debug(f"target_field: {self.target_field}")
    
    def prepare_plugin(self):
        from msserviceprofiler.modelevalstate.config.config import get_settings, field_to_param
        from msserviceprofiler.modelevalstate.config.model_config import MindieModelConfig
        from msserviceprofiler.modelevalstate.optimizer.plugins.benchmark import AisBench
        from msserviceprofiler.modelevalstate.optimizer.plugins.simulate import Simulator, DisaggregationSimulator
        # 运行默认参数服务，获取理论推导需要的指标
        if isinstance(self.scheduler.simulator, (Simulator, DisaggregationSimulator)):
            settings = get_settings()
            mc = None
            if is_mindie() and settings.theory_guided_enable:
                mc = MindieModelConfig(self.scheduler.simulator.config.config_path)
            for _, _field in enumerate(self.target_field):
                if _field.config_position.startswith("BackendConfig"):
                    _field.value = get_required_field_from_json(self.scheduler.simulator.default_config,
                                                                _field.config_position)
                elif _field.config_position == "env":
                    _field.value = os.getenv(_field.name, _field.value)
            self.default_run_param = field_to_param(self.target_field)
            self.default_res = self.scheduler.run(self.default_run_param, self.target_field)
            if self.default_res.generate_speed:
                self.gen_speed_target = 10 * self.default_res.generate_speed
            self.default_fitness = self.minimum_algorithm(self.default_res)
            self.scheduler.save_result(fitness=self.default_fitness)
            if is_mindie():
                self.mindie_prepare(mc)
            if isinstance(self.scheduler.benchmark, AisBench):
                #由于aisbench在过高的concurrency下可能会卡死，所以需要根据benchmark结果来设置concurrency的范围
                _concurrency = self.scheduler.benchmark.get_best_concurrency()
                for _field in self.target_field:
                    if _field.name in CONCURRENCYS and _field.min != _field.max:
                        if _field.min < _concurrency < _field.max:
                            _field.value = _field.max = _concurrency
                        elif _concurrency < _field.min:
                            _field.value = _field.max = _field.min
                        else:
                            _field.value = _field.max
        self.default_run_param = field_to_param(self.target_field)
        self.default_res = self.scheduler.run(self.default_run_param, self.target_field)
        self.default_fitness = self.minimum_algorithm(self.default_res)
        self.scheduler.save_result(fitness=self.default_fitness)
        
        if self.scheduler.error_info:
            raise ValueError(f"Failed to start the default service. "
                             "Please check if the service and the request to start it are correct. error:{e}")

        if (self.default_res.generate_speed is None or self.default_res.time_to_first_token is None or
                self.default_res.time_per_output_token is None):
            logger.warning(f"Failed to obtain benchmark metric data. metric {self.default_res}"
                             "Please check if the benchmark is running successfully. ")
        
        self.target_field = [*self.scheduler.simulator.data_field,
                             *self.scheduler.benchmark.data_field]

    def run_plugin(self):
        from msserviceprofiler.modelevalstate.config.config import map_param_with_value
        from msserviceprofiler.modelevalstate.optimizer.global_best_custom import CustomGlobalBestPSO
        self.prepare_plugin()
        # 备份原target field, 调整新的target field用来寻优
        with adapter_target_field(self), sample(self.scheduler):
            if self.load_breakpoint:
                self.load_history_data = self.scheduler.data_storage.load_history_position(
                    self.scheduler.data_storage.config.store_dir,
                    filter_field={**self.scheduler.data_storage.get_run_info(), REAL_EVALUATION: True}
                )
            if self.load_history_data and self.load_breakpoint:
                self.history_pos, self.history_cost = self.computer_fitness()
            optimizer = CustomGlobalBestPSO(n_particles=self.n_particles, dimensions=self.dimensions(),
                                            options=self.pso_options.model_dump(), bounds=self.constructing_bounds(),
                                            init_pos=self.init_pos, breakpoint_pos=self.history_pos,
                                            breakpoint_cost=self.history_cost, **self.pso_init_kwargs)
            with enable_simulate(self.scheduler):
                try:
                    cost, joint_vars = optimizer.optimize(self.op_func, iters=self.iters)
                except ValueError as e:
                    if "operands could not be broadcast together with shape" in str(e):
                        logger.warning(f"The first round of operation may have failed; please try again. error: {e}")
                        cost, joint_vars = optimizer.optimize(self.op_func, iters=self.iters)
                    else:
                        raise e
                best_results = self.scheduler.data_storage.get_best_result()
        _record_fitness, _record_params, _record_res = self.refine_optimization_candidates(best_results)
        best_fitness, best_param, best_performance_index = self.best_params(_record_fitness, _record_params,
                                                                _record_res)
        if best_param is None or best_fitness is None or best_performance_index is None:
            return
        _position = {_field.name: _field.value for _field in map_param_with_value(best_param, self.target_field)}
        logger.debug(f"vars: {_position}, performance index: "
                    f"ttft: {best_performance_index.time_to_first_token} \n"
                    f"tpot: {best_performance_index.time_per_output_token} \n"
                    f"generate_speed: {best_performance_index.generate_speed} \n")
            

@contextmanager
def adapter_target_field(pso_optimizer: PSOOptimizer):
    _bak_target_field = pso_optimizer.target_field
    target_field = deepcopy(pso_optimizer.target_field)
    for _field in target_field:
        # 将并发 和 req rate 设置为固定值，不进行pso寻优
        if _field.name in CONCURRENCYS and _field.constant is None:
            _field.constant = _field.value = _field.convert_dtype(_field.max)
        elif _field.name in REQUESTRATES and _field.constant is None:
            _field.constant = _field.convert_dtype(_field.max)
            _field.value = None
        elif _field.constant and _field.constant != _field.value:
            _field.value = _field.constant
    pso_optimizer.target_field = target_field
    yield
    # 恢复寻优参数配置
    pso_optimizer.target_field = _bak_target_field


@contextmanager
def sample(scheduler):
    from msserviceprofiler.modelevalstate.config.config import get_settings
    from msserviceprofiler.modelevalstate.optimizer.interfaces.benchmark import BenchmarkInterface
    """
    采样请求控制。当只跑部分请求时 可以这样处理。
    Args:
        scheduler: 调度进行运行器。

    Returns:

    """
    settings = get_settings()
    _bak_number = None
    if settings.sample_size:
        if (isinstance(scheduler.benchmark, BenchmarkInterface) and
                scheduler.benchmark.num_prompts and
                scheduler.benchmark.num_prompts > settings.sample_size):
            _bak_number = scheduler.benchmark.num_prompts
            scheduler.benchmark.num_prompts = settings.sample_size
        yield
        if _bak_number:
            scheduler.benchmark.num_prompts = _bak_number
    else:
        yield


@contextmanager
def enable_simulate(scheduler):
    """
    进入启动仿真模型
    Args: scheduler: 调度进行运行器
    Returns
    """
    if simulate_flag:
        with scheduler.simulator.enable_simulation_model as flag:
            yield flag
    else:
        yield False
    
    
def plugin_main(args: argparse.Namespace):
    from msserviceprofiler.modelevalstate.optimizer.server import main as slave_server
    from msserviceprofiler.modelevalstate.optimizer.store import DataStorage
    from msserviceprofiler.modelevalstate.config.config import get_settings
    from msserviceprofiler.modelevalstate.optimizer.experience_fine_tunning import FineTune
    from msserviceprofiler.modelevalstate.optimizer.scheduler import Scheduler
    from msserviceprofiler.modelevalstate.optimizer.plugins.simulate import DisaggregationSimulator
    from msserviceprofiler.modelevalstate.optimizer.register import register_ori_functions
    register_ori_functions()
    settings = get_settings()
    if is_root():
        logger.warning(
            "Security Warning: Do not run this tool as root. "
            "Running with elevated privileges may compromise system security. "
            "Use a regular user account."
        )   
    bak_path = None
    if args.backup:
        bak_path = settings.output.joinpath("bak")
        if not bak_path.exists():
            bak_path.mkdir(parents=True, mode=0o750)
    _simu = _bench = None
    _target_field = []
    if args.engine:
        if args.engine == 'mindie' and args.pd == PDPolicy.disaggregation.value:
            _simu = DisaggregationSimulator(bak_path=bak_path)
        else:
            _simu = simulates[args.engine](bak_path=bak_path)
        _target_field.extend(_simu.data_field)
    if args.benchmark_policy:
        _bench = benchmarks[args.benchmark_policy](bak_path=bak_path)
        _target_field.extend(_bench.data_field)
    _target_field = tuple(_target_field)
    if not _simu:
        raise ValueError("No available simulator object found.")
    if not _bench:
        raise ValueError("No available benchmark object found.")
    if len(_target_field) < 1:
        raise ValueError("No optimization fields were found. ")
    # 存储结果，只在主节点存储结果
    data_storage = DataStorage(settings.data_storage, _simu, _bench)
    # 初始化调度模块，支持单机和多机
    scheduler = Scheduler(_simu, _bench, data_storage, bak_path=bak_path)
    fine_tune = FineTune(ttft_penalty=settings.ttft_penalty,
                         tpot_penalty=settings.tpot_penalty,
                         target_field=_target_field,
                         ttft_slo=settings.ttft_slo,
                         tpot_slo=settings.tpot_slo,
                         slo_coefficient=settings.slo_coefficient,
                         step_size=settings.step_size)
    # set constraints
    constraint_thresholds = {
        'ttft': getattr(settings, 'ttft_slo', 1.0) * (1 + getattr(settings, 'slo_coefficient', 0)),
        'tpot': getattr(settings, 'tpot_slo', 0.05) * (1 + getattr(settings, 'slo_coefficient', 0))
    }
    try:
        pso = PSOOptimizer(scheduler,
                       n_particles=settings.n_particles,
                       iters=settings.iters,
                       target_field=_target_field,
                       ttft_penalty=settings.ttft_penalty,
                       tpot_penalty=settings.tpot_penalty,
                       success_rate_penalty=settings.success_rate_penalty,
                       ttft_slo=settings.ttft_slo,
                       tpot_slo=settings.tpot_slo,
                       success_rate_slo=settings.success_rate_slo,
                       generate_speed_target=settings.generate_speed_target,
                       load_breakpoint=args.load_breakpoint,
                       fine_tune=fine_tune,
                       max_fine_tune=settings.max_fine_tune,
                       pso_init_kwargs={"ftol": settings.ftol, "ftol_iter": settings.ftol_iter},
                       )
        pso.run_plugin()
    except Exception as e:	
        logger.error(f"Failed to run optimizer. Please check. error: {e}")


def arg_parse(subparsers):
    from msserviceprofiler.modelevalstate.plugins import load_general_plugins
    plugin = load_general_plugins()
    sims = ["vllm", "mindie"]
    benches = ["ais_bench", "vllm_benchmark"]
    parser = subparsers.add_parser(
        "optimizer", formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="optimize for performance"
    )
    parser.add_argument("-lb", "--load_breakpoint", default=False, action="store_true",
                        help="Continue from where the last optimization was aborted.")
    parser.add_argument("-d", "--deploy_policy", default=DeployPolicy.single.value,
                        choices=[k.value for k in list(DeployPolicy)],
                        help="Indicates whether the multi-node running policy is used.")
    parser.add_argument("--backup", default=False, action="store_true",
                        help="Whether to back up data.")
    parser.add_argument("--pd", default=PDPolicy.competition.value,
                        choices=[k.value for k in list(PDPolicy)],
                        help="whether pd competition or pd disaggregation")
    parser.add_argument("-b", "--benchmark_policy", default='ais_bench', choices=list(benchmarks.keys()) + benches,
                        help="Whether to use custom performance indicators.")
    parser.add_argument("-e", "--engine", default='mindie', choices=list(simulates.keys()) + sims,
                        help="The engine used for model evaluation.")
    parser.set_defaults(func=plugin_main)

