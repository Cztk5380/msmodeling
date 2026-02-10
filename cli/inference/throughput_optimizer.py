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
import copy
import time
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Iterator

import pandas as pd
import torch
from serving_cast.service.optimizer_factory import OptimizerFactory
from serving_cast.service.optimizer_summary import OptimizerSummary

from serving_cast.service.utils import (
    BatchRangeAction,
    check_positive_float,
    check_positive_integer,
    check_string_valid,
    LIMIT_COUNT,
    logger,
    OptimizerData,
    set_log_level,
)

from tensor_cast.core.model_runner import ModelRunner

from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile


class ParallelRunner:
    def __init__(self, args):
        self.args = args
        self.user_input = UserInputConfig.from_args(args)

        self.summary_result = []
        self.device_profile = DeviceProfile.all_device_profiles[self.args.device]
        if self.device_profile.comm_grid.grid.nelement() < self.args.num_devices:
            raise ValueError(
                f"No communication grid found for {self.args.num_devices} devices."
            )
        self.optimizer_data = OptimizerData(
            input_length=self.args.input_length,
            output_length=self.args.output_length,
            ttft_limits=self.args.ttft_limits,
            max_prefill_tokens=self.args.max_prefill_tokens,
            num_devices=self.args.num_devices,
            serving_cost=self.args.serving_cost,
            num_mtp_tokens=self.args.num_mtp_tokens,
            mtp_acceptance_rate=self.args.mtp_acceptance_rate,
        )

    def run_agg(self) -> list[OptimizerSummary]:
        logger.info(
            "Run Aggregation with ttft %r ms, tpot %r ms.",
            self.args.ttft_limits,
            self.args.tpot_limits,
        )
        overwrite_optimizer_data = copy.deepcopy(self.optimizer_data)
        overwrite_optimizer_data.tpot_limits = self.args.tpot_limits
        df_list = self._get_df_list(overwrite_optimizer_data)

        self._add_summary_result(df_list, overwrite_optimizer_data)

        return self.summary_result

    def run_disagg(self) -> list[OptimizerSummary]:
        # if set ttft_limits, run Prefill; if set tpot_limits, run Decode
        if self.args.ttft_limits is not None:
            logger.info("Run Prefill with ttft %r ms.", self.args.ttft_limits)
            overwrite_optimizer_data = copy.deepcopy(self.optimizer_data)
            overwrite_optimizer_data.ttft_limits = self.args.ttft_limits
            overwrite_optimizer_data.tpot_limits = None
            df_list = self._get_df_list(overwrite_optimizer_data)
            self._add_summary_result(df_list, overwrite_optimizer_data)

        if self.args.tpot_limits is not None:
            logger.info("Run Decode with tpot %r ms.", self.args.tpot_limits)
            overwrite_optimizer_data = copy.deepcopy(self.optimizer_data)
            overwrite_optimizer_data.tpot_limits = self.args.tpot_limits
            overwrite_optimizer_data.ttft_limits = None
            df_list = self._get_df_list(overwrite_optimizer_data)
            self._add_summary_result(df_list, overwrite_optimizer_data)

        return self.summary_result

    def _add_summary_result(
        self, df_list: list[pd.DataFrame], overwrite_data_config: OptimizerData
    ):
        if len(df_list) == 0:
            logger.info(
                "No results found with ttft %r ms, tpot %r ms",
                overwrite_data_config.ttft_limits,
                overwrite_data_config.tpot_limits,
            )
            return
        summary = OptimizerSummary(overwrite_data_config)
        summary.set_summary_df(pd.concat(df_list, axis=0, ignore_index=True))
        self.summary_result.append(summary)

    def _get_model_runnner(self, user_input: UserInputConfig) -> ModelRunner:
        model_runner = None
        try:
            model_runner = ModelRunner(user_input)
        except Exception:
            logger.error("Failed to build model %r", self.args.model_id)

        return model_runner

    def _get_user_config(self) -> Iterator[UserInputConfig]:
        # get tp list
        tp_list = getattr(self.args, "tp_sizes", None)
        if tp_list is None:
            tp_list = [1 << i for i in range(self.args.num_devices.bit_length())]

        for tp in tp_list:
            tmp_user_input = copy.deepcopy(self.user_input)
            tmp_user_input.tp_size = tp
            # if the moe_config is None, ep will be set False in update_parallel_config
            # so set it True here, moe models can enable ep parallel correctly
            tmp_user_input.ep_size = tmp_user_input.world_size
            tmp_user_input.moe_dp_size = 1
            tmp_user_input.moe_tp_size = 1
            if self.args.num_devices % tp != 0:
                continue
            yield tmp_user_input

    def _get_df_list(
        self, overwrite_optimizer_data: OptimizerData
    ) -> list[pd.DataFrame]:
        df_list = []

        with ProcessPoolExecutor(max_workers=self.args.jobs) as executor:
            future_to_config = {
                executor.submit(
                    self._submit_task,
                    user_config,
                    overwrite_optimizer_data,
                ): user_config
                for user_config in self._get_user_config()
            }

            for future in as_completed(future_to_config):
                result_df = future.result()
                if result_df is not None:
                    df_list.append(result_df)

        return df_list

    def _submit_task(
        self, user_input: UserInputConfig, overwrite_optimizer_data: OptimizerData
    ):
        # 1. get model config
        if self.args.compile:
            torch._dynamo.config.recompile_limit = LIMIT_COUNT
            torch._dynamo.config.accumulated_recompile_limit = LIMIT_COUNT
        torch.compiler.reset()
        logger.info("Start processing TP size: %d", user_input.tp_size)
        model_runner = self._get_model_runnner(user_input)
        if model_runner is None:
            return None
        if (
            model_runner.model.model_config.mla_config is None
            and model_runner.model.text_config.num_key_value_heads
            % model_runner.model.model_config.parallel_config.tensor_parallel_size
            != 0
        ):
            logger.warning(
                "No MLA or TEXT config found for model %r, skip.", self.args.model_id
            )
            return None
        # 2. get strategy result
        strategy = OptimizerFactory.create_strategy(model_runner, self.args.disagg)
        result = strategy.run(overwrite_optimizer_data, self.args.batch_range)
        if (
            not isinstance(result, OptimizerSummary)
            or len(result.get_summary_df()) == 0
        ):
            logger.warning(
                "No result found with TP %d for ttft %s ms, tpot %s ms",
                model_runner.model.model_config.parallel_config.tensor_parallel_size,
                overwrite_optimizer_data.ttft_limits,
                overwrite_optimizer_data.tpot_limits,
            )
            return None
        result_df = result.get_summary_df()
        logger.info(
            "Finish processing TP size: %d",
            model_runner.model.model_config.parallel_config.tensor_parallel_size,
        )

        return result_df


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Get Best Throughput for given input/output sequence length and SLO limitations "
        "in aggregation mode or disaggregation mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-length",
        type=check_positive_integer,
        required=True,
        help="The input length of the prompt.",
    )
    parser.add_argument(
        "--output-length",
        type=check_positive_integer,
        required=True,
        help="The expected output length.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=list(DeviceProfile.all_device_profiles.keys()),
        help="The device type for benchmarking.",
    )
    parser.add_argument(
        "--model-id",
        type=check_string_valid,
        required=True,
        help="Model ID from Hugging Face (e.g., 'meta-llama/Llama-2-7b-hf').",
    )
    parser.add_argument(
        "--num-devices",
        type=check_positive_integer,
        default=1,
        help="Number of devices",
    )
    model_group = parser.add_argument_group("Model & Quantization Options")
    model_group.add_argument(
        "--compile",
        action="store_true",
        help="If set, invoke torch.compile() on the model before inference.",
    )
    model_group.add_argument(
        "--compile-allow-graph-break",
        action="store_true",
        help="If set, invoke torch.compile() on the model before inference.",
    )
    model_group.add_argument(
        "--num-mtp-tokens",
        type=int,
        choices=range(0, 10),
        default=0,
        help="Number of MTP tokens, 0 means disabled - only support models having MTP like DeepSeek",
    )
    parser.add_argument(
        "--mtp-acceptance-rate",
        type=float,
        default=[0.9, 0.6, 0.4, 0.2],
        nargs="+",
        help="Acceptance rate list for MTP",
    )
    model_group.add_argument(
        "--quantize-linear-action",
        type=QuantizeLinearAction,
        choices=list(QuantizeLinearAction),
        default=QuantizeLinearAction.W8A8_DYNAMIC,
        help="Quantize all linear layers in the model from choices (currently only support symmetric quant)",
    )
    model_group.add_argument(
        "--mxfp4-group-size",
        type=check_positive_integer,
        default=32,
        help="Group size for MXFP4 quantization",
    )
    model_group.add_argument(
        "--quantize-attention-action",
        type=QuantizeAttentionAction,
        choices=list(QuantizeAttentionAction),
        default=QuantizeAttentionAction.DISABLED,
        help="Quantize the KV cache with the given action",
    )
    model_group.add_argument(
        "--reserved-memory-gb",
        type=float,
        default=0,
        help="Size of reserved device memory (in GB) that we cannot use from applications.",
    )
    model_group.add_argument(
        "--tp-sizes",
        type=int,
        nargs="+",
        default=None,
        help="TP sizes to search (default: powers of 2 up to world_size)",
    )
    service_group = parser.add_argument_group("Service Options")
    service_group.add_argument(
        "--ttft-limits",
        type=check_positive_float,
        default=None,
        help="TTFT constraints under which to search for the best throughput. None means no constraint.",
    )
    service_group.add_argument(
        "--tpot-limits",
        type=check_positive_float,
        default=None,
        help="TPOT constraints under which to search for the best throughput. None means no constraint.",
    )
    service_group.add_argument(
        "--max-prefill-tokens",
        type=check_positive_integer,
        default=8192,
        help="Max prefill tokens",
    )
    service_group.add_argument(
        "--batch-range",
        type=int,
        nargs="+",
        action=BatchRangeAction,
        default=None,
        help="Batch size range: [min max] or [max] (default: 1 for min, no limit for max)",
    )
    service_group.add_argument(
        "--serving-cost",
        type=float,
        default=0,
        help="Serving cost represents the cost of service delivery",
    )
    service_group.add_argument(
        "--disagg",
        action="store_true",
        help="If set, run disaggregation mode. disagg means disaggregation mode.",
    )
    service_group.add_argument(
        "--jobs",
        type=check_positive_integer,
        default=8,
        help="Number of parallel jobs.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level to print",
    )
    parser.add_argument(
        "--dump-original-results",
        action="store_true",
        help="If set, dump the original results for analysis.",
    )
    args = parser.parse_args()
    return args


def main():
    start_time = time.time()
    args = arg_parse()
    set_log_level(args.log_level)
    if args.max_prefill_tokens < args.input_length:
        logger.warning(
            "max_prefill_tokens (%r) is smaller than input_length (%r). "
            "We currently do not have support for this scenario.",
            args.max_prefill_tokens,
            args.input_length,
        )
        return
    if (
        args.num_mtp_tokens > 0
        and args.num_mtp_tokens > len(args.mtp_acceptance_rate) + 1
    ):
        logger.warning(
            "num_mtp_tokens (%r) is greater than the length of mtp_acceptance_rate (%r). Please check.",
            args.num_mtp_tokens,
            len(args.mtp_acceptance_rate),
        )
        return
    logger.info("Starting experiments.")
    tasks = ParallelRunner(args)
    if args.disagg:
        results = tasks.run_disagg()
    else:
        results = tasks.run_agg()
    for res in results:
        res.report_final_result(args)
    end_time = time.time()
    logger.info("All experiments completed in %.2f seconds.", end_time - start_time)


if __name__ == "__main__":
    main()
