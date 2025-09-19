from typing import Dict, Optional

from ..model_config import (
    MoEConfig,
    MoEFieldNames,
    RepetitiveLayerConfig,
    RepetitiveRange,
)

# TODO: Allow users to extend these default configurations from config.py


_model_id_to_json_tbl: dict[str, str] = {
    "moonshotai/Kimi-K2-Base": "kimi_k2.json",
    "deepseek-ai/DeepSeek-V3.1": "deepseek_v3.1.json",
}


def model_id_to_json(model_id: str) -> Optional[str]:
    return _model_id_to_json_tbl.get(model_id)


# model_id -> MoEConfig
_model_id_to_moe_config: Dict[str, MoEConfig] = {
    "zai-org/GLM-4.5": MoEConfig(
        module_name="Glm4MoeMoE",
    ),
    "Qwen/Qwen3-235B-A22B": MoEConfig(
        module_name="Qwen3MoeSparseMoeBlock",
        gate_returns_raw_logits=True,
    ),
    "Qwen/Qwen3-Next-80B-A3B-Instruct": MoEConfig(
        module_name="Qwen3NextSparseMoeBlock",
        gate_returns_raw_logits=True,
        field_names=MoEFieldNames(
            shared_experts="shared_expert", shared_experts_gate="shared_expert_gate"
        ),
    ),
    "moonshotai/Kimi-K2-Base": MoEConfig(
        module_name="DeepseekV3MoE",
    ),
    "deepseek-ai/DeepSeek-V3.1": MoEConfig(
        module_name="DeepseekV3MoE",
    ),
}


def model_id_to_moe_config(model_id: str) -> Optional[MoEConfig]:
    return _model_id_to_moe_config.get(model_id)


_model_id_to_mla_module_name: Dict[str, str] = {
    "deepseek-ai/DeepSeek-V3.1": "DeepseekV3Attention",
    "moonshotai/Kimi-K2-Base": "DeepseekV3Attention",
}


def model_id_to_mla_module_name(model_id: str):
    return _model_id_to_mla_module_name.get(model_id)


_model_id_to_mtp_block_module_name: Dict[str, str] = {
    "deepseek-ai/DeepSeek-V3.1": "DeepseekV3DecoderLayer",
    "moonshotai/Kimi-K2-Base": "DeepseekV3DecoderLayer",
}


def model_id_to_mtp_block_module_name(model_id: str) -> str:
    return _model_id_to_mtp_block_module_name.get(model_id)


def strip_module_name(name: str) -> str:
    """Strip `_inner` module name from the given module path name"""
    stripped = name.removeprefix("_inner.")
    return stripped.replace("._inner.", ".")


_model_id_to_repetition_config: Dict[str, RepetitiveLayerConfig] = {
    "deepseek-ai/DeepSeek-V3.1": RepetitiveLayerConfig(
        repetitive_ranges=[
            RepetitiveRange(0, 1, 3),  # Dense
            RepetitiveRange(3, 4, -1),  # Sparse
        ],
    ),
    "moonshotai/Kimi-K2-Base": RepetitiveLayerConfig(
        repetitive_ranges=[
            RepetitiveRange(0, 1, 1),  # Dense
            RepetitiveRange(1, 2, -1),  # Sparse
        ],
    ),
}


def model_id_to_repetition_config(model_id: str) -> RepetitiveLayerConfig:
    return _model_id_to_repetition_config.get(model_id)


def default_repetition_config() -> RepetitiveLayerConfig:
    return RepetitiveLayerConfig(
        repetitive_ranges=[
            RepetitiveRange(0, 1, -1),
        ],
    )
