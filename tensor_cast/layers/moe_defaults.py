from typing import Dict

from ..model_config import MoEConfig

# TODO: perhaps we'd better move these configurations to some
#       global configuration file that can be easily found and
#       configured by users.

# model_id -> MoEConfig
model_id_to_config: Dict[str, MoEConfig] = {
    "zai-org/GLM-4.5": MoEConfig(
        module_name="Glm4MoeMoE",
    ),
    "Qwen/Qwen3-235B-A22B": MoEConfig(
        module_name="Qwen3MoeSparseMoeBlock",
        gate_returns_raw_logits=True,
    ),
}
