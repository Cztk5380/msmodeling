from typing import Dict, Optional

from ..layers.mla import MultiheadLatentAttentionBase

from ..model_config import AttentionQuantConfig, MoEConfig, MoEFieldNames

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
    "Qwen/Qwen3-30B-A3B": MoEConfig(
        module_name="Qwen3MoeSparseMoeBlock",
        gate_returns_raw_logits=True,
    ),
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": MoEConfig(
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
    stripped_before = name
    while stripped != stripped_before:
        stripped_before = stripped
        stripped = stripped_before.removeprefix("_inner.")
    stripped = stripped.replace("._inner.", ".")
    stripped_before = stripped
    stripped = stripped_before.removesuffix("._inner")
    while stripped != stripped_before:
        stripped_before = stripped
        stripped = stripped_before.removesuffix("._inner")
    return stripped


def get_attention_quant_config(model, layer_idx) -> Optional[AttentionQuantConfig]:
    if model.model_config.mla_config is not None:
        for _, module in model._inner.named_modules():
            if (
                isinstance(module, MultiheadLatentAttentionBase)
                and hasattr(module, "layer_idx")
                and module.layer_idx == layer_idx
                and (attn_quant_config := module.quant_config) is not None
            ):
                return attn_quant_config
    if layer_idx in model.attention_by_layers:
        return model.attention_by_layers[layer_idx].quant_config
    return None
