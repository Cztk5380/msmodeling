# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import List, Optional, Dict

import stime
from service_sim.device import Device
from service_sim.request import Request, RequestState
from tensor_cast.interface.model_runner import ModelRunner as TensorCastModelRunner
from service_sim.config import Config

logger = stime.get_logger(__name__)


class ModelRunner:
    def __init__(self, parallel_config, world_size, device_type, dp_rank: int):
        """
        Each model runner works on a data-parallel partition ('dp_rank') given a list of devices
        to compute the model given the model configuration 'model_config'. The computation is
        further sharded among the 'devices' via other parallel algorithms like tensor parallel etc.
        It instantiates a list of 'Workers' with each working on a device.
        """
        self.common_config = Config.get_instance().common_config
        self.parallel_config = parallel_config
        self.inference_engine = TensorCastModelRunner(
            device=device_type,
            model_id=self.common_config.model_config.name,
            do_compile=self.common_config.model_config.do_compile,
            allow_graph_break=self.common_config.model_config.allow_graph_break,
            dump_input_shapes=self.common_config.model_config.dump_input_shapes,
            chrome_trace=self.common_config.model_config.chrome_trace,
            quantize_linear_action=self.common_config.model_config.quantize_linear_action,
            quantize_lmhead=self.common_config.model_config.quantize_lmhead,
            mxfp4_group_size=self.common_config.model_config.mxfp4_group_size,
            quantize_attention_action=self.common_config.model_config.quantize_attention_action,
            num_mtp_tokens=self.common_config.model_config.num_mtp_tokens,
            num_hidden_layers_override=0,
            world_size=world_size,
            tp_size=self.parallel_config.tp_size,
            dp_size=self.parallel_config.dp_size,
            mlp_tp_size=self.parallel_config.mlp_tp_size,
            mlp_dp_size=self.parallel_config.mlp_dp_size,
            lmhead_tp_size=self.parallel_config.lmhead_tp_size,
            lmhead_dp_size=self.parallel_config.lmhead_dp_size,
            ep=self.parallel_config.ep,
            reserved_memory_gb=0.0,
            block_size=self.common_config.serving_config.block_size,
        )


    @staticmethod
    def request2dict(batch: List[Request]) -> List[Dict[str, int]]:
        transferred_batch = []
        for req in batch:
            query_len = req.query_len
            seq_len = req.seq_len
            if not query_len <= seq_len:
                raise ValueError(f"req_id: {req.id}, query_len {query_len} > seq_len {seq_len}")
            if req.state not in [RequestState.PREFILLING, RequestState.DECODING]:
                raise ValueError(f"req_id: {req.id}, state {req.state} is not PREFILLING or DECODING")
            is_decode = req.state == RequestState.DECODING
            transferred_batch.append({"query_len": query_len, "seq_len": seq_len, "is_decode": is_decode})

        return transferred_batch

    def process_batch(self, batch: List[Request]):
        batch = self.request2dict(batch)
        duration = self._get_estimated_time(batch)
        with stime.Duration(duration):
            logger.debug(
                f"{self.common_config.model_config.name} process batch, batch length: {len(batch)}, "
                f"consume {duration} seconds"
            )
    
    def warmup(self) -> int:
        '''
        use max length to try warmup get num_blocks
        '''
        serving_config = Config.get_instance().common_config.serving_config
        batch = [
            {
                "seq_len": serving_config.max_tokens_budget,
                "query_len": serving_config.max_tokens_budget,
                "is_decode": False
            }
        ]
        inference_metrics = self.inference_engine.run_inference(batch)
        block_size = self.common_config.serving_config.block_size
        all_mem_for_kv_cache = inference_metrics.device_memory_available_gb + inference_metrics.kv_cache_size_gb
        num_blocks = int(all_mem_for_kv_cache / inference_metrics.kv_cache_per_token_gb // block_size)
        logger.debug(f"warmup result: {num_blocks} blocks")
        return num_blocks, block_size

    def _get_estimated_time(self, batch):
        estimated_time = self.inference_engine.run_inference(batch).execution_time_s
        return estimated_time

