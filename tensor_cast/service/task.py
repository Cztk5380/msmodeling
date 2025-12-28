# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

import copy
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Generator

import pandas as pd
import torch

from tensor_cast.core.utils import build_model, UserInputConfig
from tensor_cast.device import DeviceProfile
from tensor_cast.service.backend_factory import StrategyFactory
from tensor_cast.service.report_and_save import Summary
from tensor_cast.service.utils import DataConfig, LIMIT_TIME, logger


class TaskRunner:
    def __init__(self, args):
        self.args = args
        self.model_id = self.args.model_id
        self.device = self.args.device
        self.compile = self.args.compile
        self.compile_allow_graph_break = self.args.compile_allow_graph_break
        self.num_mtp_tokens = self.args.num_mtp_tokens
        self.quantize_linear_action = self.args.quantize_linear_action
        self.mxfp4_group_size = self.args.mxfp4_group_size
        self.quantize_attention_action = self.args.quantize_attention_action
        self.backend = self.args.backend
        self.num_devices = self.args.num_devices
        self.max_prefill_tokens = self.args.max_prefill_tokens
        self.user_input = UserInputConfig.from_args(args)

        self.summary_result = []
        self.device_profile = DeviceProfile.all_device_profiles[self.device]
        self.data_config = DataConfig(
            input_length=self.args.input_length,
            output_length=self.args.output_length,
            ttft_limits=self.args.ttft_limits,
            max_prefill_tokens=self.args.max_prefill_tokens,
            device_nums=self.num_devices,
            device_profile=self.device_profile,
        )
        if self.compile:
            torch._dynamo.config.recompile_limit = LIMIT_TIME
            torch._dynamo.config.accumulated_recompile_limit = LIMIT_TIME

    def run(self):
        if self.device_profile.comm_grid.grid.nelement() < self.num_devices:
            logger.warning(
                "No communication grid found for %r devices, skip.", self.num_devices
            )
            return self.summary_result
        tpot_list = (
            self.args.tpot_limits
            if isinstance(self.args.tpot_limits, list)
            else [self.args.tpot_limits]
        )
        # get paral config
        for tpot in tpot_list:
            overwrite_data_config = copy.deepcopy(self.data_config)
            overwrite_data_config.tpot_limits = tpot
            df_list = []
            summary = Summary(overwrite_data_config)

            with ProcessPoolExecutor(max_workers=12) as executor:
                future_to_config = {
                    executor.submit(
                        self._process_parallel_config,
                        user_config,
                        overwrite_data_config,
                    ): user_config
                    for user_config in self._get_user_config()
                }

                for future in as_completed(future_to_config):
                    result_df = future.result()
                    if result_df is not None:
                        df_list.append(result_df)
            if len(df_list) == 0:
                logger.info(
                    "No results found with ttft %r, tpot %rms",
                    self.args.ttft_limits,
                    tpot,
                )
                continue
            summary.set_summary_df(pd.concat(df_list, axis=0, ignore_index=True))
            self.summary_result.append(summary)

        return self.summary_result

    def _get_model(self, user_input: UserInputConfig):
        torch.compiler.reset()
        model = None
        try:
            model = build_model(user_input).eval()
        except Exception:
            logger.error("Failed to build model %r", self.model_id)

        return model

    def _get_user_config(self) -> Generator[UserInputConfig]:
        default_tp_list = [1 << i for i in range(self.num_devices.bit_length())]
        # get tp list
        tp_list = getattr(self.args, "tp", None)
        if not tp_list:
            tp_list = default_tp_list
        elif isinstance(tp_list, int):
            tp_list = [tp_list]

        for tp in tp_list:
            tmp_user_input = copy.deepcopy(self.user_input)
            tmp_user_input.tp_size = tp
            # TODO num_devices is frozen，do not need to change every time
            tmp_user_input.world_size = self.num_devices
            yield tmp_user_input

    def _process_parallel_config(
        self, user_input: UserInputConfig, overwrite_data_config
    ):
        # 1. get model config
        logger.info("Start processing TP size: %d", user_input.tp_size)
        model = self._get_model(user_input)
        if model is None:
            return None
        if (
            model.model_config.mla_config is None
            and model.text_config.num_key_value_heads
            % model.model_config.parallel_config.tensor_parallel_size
            != 0
        ):
            logger.warning(
                "No MLA or TEXT config found for model %r, skip.", self.model_id
            )
            return None
        # 2. get backend result
        backend = StrategyFactory.create_backend(self.backend, self.args, model)
        result = backend.find_best_result_under_constraints(overwrite_data_config)
        if not isinstance(result, Summary) or len(result.get_summary_df()) == 0:
            logger.warning(
                "No result found with TP %d for tpot %sms",
                model.model_config.parallel_config.tensor_parallel_size,
                overwrite_data_config.tpot_limits,
            )
            return None
        result_df = result.get_summary_df()
        logger.info(
            "Finish processing TP size: %d",
            model.model_config.parallel_config.tensor_parallel_size,
        )

        return result_df
