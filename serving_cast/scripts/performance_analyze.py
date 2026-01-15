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
import time

from serving_cast.service.task import TaskRunner
from serving_cast.service.utils import (
    BackendName,
    check_positive_float,
    check_positive_integer,
    check_string_valid,
    logger,
    set_log_level,
)

from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)
from tensor_cast.device import DeviceProfile


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
        choices=range(0, 5),
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
    service_group = parser.add_argument_group("Service Options")
    service_group.add_argument(
        "--ttft-limits",
        type=check_positive_float,
        default=float("inf"),
        help="TTFT constraints under which to search for the best throughput. inf means no constraint.",
    )
    service_group.add_argument(
        "--tpot-limits",
        type=float,
        default=[50.0],
        nargs="+",
        help="A list of TPOT constraints under which to search for the best throughput.",
    )
    service_group.add_argument(
        "--backend",
        type=check_string_valid,
        default=BackendName.MindIE.value,
        choices=[backend.value for backend in BackendName],
        help="Backend name.",
    )
    service_group.add_argument(
        "--max-prefill-tokens",
        type=check_positive_integer,
        default=8192,
        help="Max prefill tokens",
    )
    service_group.add_argument(
        "-disagg",
        "--disaggregation",
        action="store_true",
        help="If set, run in disaggregation mode.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level to print",
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
    tasks = TaskRunner(args)
    results = tasks.run()
    for res in results:
        res.report_final_result(args)
    end_time = time.time()
    logger.info("All experiments completed in %.2f seconds.", end_time - start_time)


if __name__ == "__main__":
    main()
