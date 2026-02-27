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
import logging
import time

from serving_cast.service.utils import (
    BatchRangeAction,
    check_positive_float,
    check_positive_integer,
)

from tensor_cast.core.quantization.datatypes import (
    QuantizeAttentionAction,
    QuantizeLinearAction,
)

from ..utils import get_common_argparser, LOG_FORMAT, LOG_LEVELS


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Get Best Throughput for given input/output sequence length and SLO limitations "
        "in aggregation mode or disaggregation mode.",
        parents=[get_common_argparser()],
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
        "--dump-original-results",
        action="store_true",
        help="If set, dump the original results for analysis.",
    )
    multimodal_group = parser.add_argument_group("MultiModal Options")
    multimodal_group.add_argument(
        "--image-height",
        type=check_positive_integer,
        default=None,
        help="Height of the input images",
    )
    multimodal_group.add_argument(
        "--image-width",
        type=check_positive_integer,
        default=None,
        help="Width of the input images",
    )
    args = parser.parse_args()
    return args


def main():
    start_time = time.time()
    args = arg_parse()
    logging.basicConfig(
        level=LOG_LEVELS[args.log_level.lower()],
        format=LOG_FORMAT,
    )
    logger = logging.getLogger(__name__)

    if args.max_prefill_tokens < args.input_length:
        logger.warning(
            "max_prefill_tokens (%r) is smaller than input_length (%r). "
            "We currently do not have support for this scenario.",
            args.max_prefill_tokens,
            args.input_length,
        )
        return 1

    if (
        args.num_mtp_tokens > 0
        and args.num_mtp_tokens > len(args.mtp_acceptance_rate) + 1
    ):
        logger.warning(
            "num_mtp_tokens (%r) is greater than the length of mtp_acceptance_rate (%r). Please check.",
            args.num_mtp_tokens,
            len(args.mtp_acceptance_rate),
        )
        return 1

    from serving_cast.parallel_runner import ParallelRunner

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
