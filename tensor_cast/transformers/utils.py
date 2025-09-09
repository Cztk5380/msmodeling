from typing import Dict, Optional

from liuren_modeling.tensor_cast.model_config import MoEConfig

# TODO: Allow users to extend these default configurations from config.py


_model_id_to_json_tbl: dict[str, str] = {
    "moonshotai/Kimi-K2-Base": "kimi_k2.json",
    "deepseek-ai/DeepSeek-V3.1": "deepseek_v3.1.json",
}


def model_id_to_json(model_id: str) -> Optional[str]:
    return _model_id_to_json_tbl.setdefault(model_id, None)


# model_id -> MoEConfig
_model_id_to_moe_config: Dict[str, MoEConfig] = {
    "zai-org/GLM-4.5": MoEConfig(
        module_name="Glm4MoeMoE",
    ),
    "Qwen/Qwen3-235B-A22B": MoEConfig(
        module_name="Qwen3MoeSparseMoeBlock",
        gate_returns_raw_logits=True,
    ),
    "moonshotai/Kimi-K2-Base": MoEConfig(
        module_name="DeepseekV3MoE",
    ),
    "deepseek-ai/DeepSeek-V3.1": MoEConfig(
        module_name="DeepseekV3MoE",
    ),
}


def model_id_to_moe_config(model_id: str) -> Optional[MoEConfig]:
    return _model_id_to_moe_config.setdefault(model_id, None)
