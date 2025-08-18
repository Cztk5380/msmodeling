from typing import Dict
from ..model_config import MoEFieldNames

# TODO: perhaps we'd better move these configurations to some
#       global configuration file that can be easily found and
#       configured by users.

# model_id -> MoE module name
moe_modules: Dict[str, str] = {
    "zai-org/GLM-4.5": "Glm4MoeMoE",
    "Qwen/Qwen3-235B-A22B": "Qwen3MoeSparseMoeBlock",
}

# model_id -> MoE field names, overriding the default one
moe_field_names_overrides: Dict[str, MoEFieldNames] = {
    # something like below but only needed if some fields are
    # different from the default.
    # "zai-org/GLM-4.5": MoEFieldNames(),
}
