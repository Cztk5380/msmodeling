import argparse
import copy
import logging
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import Callable, Iterator, Optional, Type

import pandas as pd
import torch

from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import DeviceProfile
from .service.optimizer_factory import OptimizerFactory
from .service.optimizer_summary import OptimizerSummary

from .service.utils import LIMIT_COUNT, OptimizerData


logger = logging.getLogger(__name__)


class ParallelRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        executor_class: Optional[Type[Executor]] = None,
        worker_initializer: Optional[Callable] = None,
    ) -> None:
        """Initializes the optimizer with device configuration and execution backend.

        This constructor sets up the device profile based on the provided configuration,
        validates that the hardware topology supports the requested number of devices,
        and prepares the parallel execution strategy.

        Args:
            config: The parsed configuration object containing run parameters
                (e.g., device type, number of devices, input/output lengths).
                Usually an argparse.Namespace.
            executor_class: A class reference used to spawn parallel workers.
                Defaults to `concurrent.futures.ProcessPoolExecutor` if not provided.
                Useful for injecting mocks during testing.
            worker_initializer: A function to run at the start of each worker process
                (e.g., for logging setup). Defaults to `self._init_worker`.
                Must be picklable.

        Raises:
            ValueError: If the available communication grid in the device profile
                cannot support the requested number of devices (`num_devices`).
        """
        self.args = args
        self.device_profile = DeviceProfile.all_device_profiles[self.args.device]
        if self.device_profile.comm_grid.grid.nelement() < self.args.num_devices:
            raise ValueError(
                f"No communication grid found for {self.args.num_devices} devices."
            )

        self._executor_class = executor_class if executor_class else ProcessPoolExecutor
        self._worker_initializer = (
            worker_initializer if worker_initializer else self._init_worker
        )

        self.user_input = UserInputConfig.from_args(args)
        self.summary_result = []
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
        with self._executor_class(
            max_workers=self.args.jobs, initializer=self._worker_initializer
        ) as executor:
            # use partial to make sure it is serializable
            results = executor.map(
                partial(
                    self._submit_task, overwrite_optimizer_data=overwrite_optimizer_data
                ),
                self._get_user_config(),
            )

            try:
                return [r for r in results if r is not None]
            except BrokenProcessPool:
                logger.error(
                    "A worker process crashed unexpectedly during execution. "
                    "Common causes: memory issues, unpicklable objects, or unhandled exceptions in worker."
                )
                logger.error(
                    "Executor: %s, Workers: %s",
                    self._executor_class.__name__,
                    self.args.jobs,
                )
                logger.error("Worker initializer: %s", self._worker_initializer)
                raise

    def _init_worker(self) -> None:
        """Initialize logging configuration for worker processes.

        This method is called when each worker process starts in a ProcessPoolExecutor.
        It reconfigures the logging system with the same settings as the main process
        to ensure consistent logging behavior across all processes.

        The logging configuration includes:
        - Log level: Taken from command-line argument (converted to uppercase)
        - Format: Fixed format string showing level, logger name, and message

        Note:
            This is necessary because multiprocessing creates separate processes
            that do not inherit the parent process's logging configuration.
            Each worker must explicitly reconfigure logging.
        """
        log_level_name = self.args.log_level.upper()
        log_level = logging._nameToLevel[log_level_name]

        logging.basicConfig(
            level=log_level, format="[%(levelname)s] [%(name)s] %(message)s"
        )

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
