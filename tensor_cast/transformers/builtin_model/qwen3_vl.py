import torch

from tensor_cast.layers import COLWISE_LINEAR, ROWWISE_LINEAR
from tensor_cast.transformers.custom_model_registry import (
    ModelProfile,
    register_model_profile,
    resolve_visual_config,
)


QWEN3_VL_VISUAL_CONFIG = resolve_visual_config(
    {
        "visual_merger_linear_mapping": {
            "visual.merger.linear_fc1": COLWISE_LINEAR,
            "visual.merger.linear_fc2": ROWWISE_LINEAR,
            "visual.deepstack_merger_list.*.linear_fc1": COLWISE_LINEAR,
            "visual.deepstack_merger_list.*.linear_fc2": ROWWISE_LINEAR,
        },
        "visual_mlp_linear_mapping": {
            "visual.blocks.*.mlp.linear_fc1": COLWISE_LINEAR,
            "visual.blocks.*.mlp.linear_fc2": ROWWISE_LINEAR,
        },
    }
)


def patch_method_for_qwen3_vl():
    """
    Patch the Qwen3-VL model to fix simulation issues in meta mode.
      Problem background:
      1. The Qwen3-VL model uses boolean-mask-based tensor indexing operations
        (e.g., inputs_embeds[special_image_mask], hidden_states[visual_pos_masks, :]).
      2. These operations cannot run correctly in meta mode because:
         * They internally call nonzero(), whose output shape depends on actual values and
           cannot be inferred in meta mode.
         * Even with meta_nonzero_assume_all_nonzero enabled, dimension mismatch errors still occur.

      Solution:
      * Skip tensor count validation in get_placeholder_mask.
      * Skip the deep stack fusion logic in _deepstack_process.
    """

    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLTextModel,
    )
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeModel,
        Qwen3VLMoeTextModel,
    )

    # Class to be patched
    TARGET_CLASSES = [Qwen3VLModel, Qwen3VLMoeModel]
    # Save the original method of each class.
    ORIGINAL_METHODS = {cls: cls.get_placeholder_mask for cls in TARGET_CLASSES}

    def patched_get_placeholder_mask(self, *args, **kwargs):
        # Forcibly skip image_features
        kwargs["image_features"] = None
        # Invoke the original method of the corresponding class.
        return ORIGINAL_METHODS[type(self)](self, *args, **kwargs)

    for cls in TARGET_CLASSES:
        cls.get_placeholder_mask = patched_get_placeholder_mask

    DEEPSTACK_PROCESS_TARGET_CLASSES = [Qwen3VLTextModel, Qwen3VLMoeTextModel]

    def _patched_deepstack_process(
        self, hidden_states, visual_pos_masks, visual_embeds
    ):
        return hidden_states

    for cls in DEEPSTACK_PROCESS_TARGET_CLASSES:
        cls._deepstack_process = _patched_deepstack_process


class TensorQwen3VLMoeTextMLP(torch.nn.Module):
    def __init__(self, original_module: torch.nn.Module):
        super().__init__()
        self.hidden_size = original_module.hidden_size
        self.intermediate_size = original_module.intermediate_size
        self.act_fn = original_module.act_fn
        # Split gate_up_proj into separate gate_proj and up_proj for proper TP sharding
        self.gate_proj = torch.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = torch.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = torch.nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(up * self.act_fn(gate))
        return hidden_states


register_model_profile(
    ModelProfile(
        model_type="qwen3_vl_moe",
        moe_module_name="Qwen3VLMoeTextSparseMoeBlock",
        moe_gate_returns_raw_logits=True,
        moe_num_experts_key=["text_config", "num_experts"],
        custom_expert_module_type=TensorQwen3VLMoeTextMLP,
        model_family="qwen3_vl",
        vl_patch_method=patch_method_for_qwen3_vl,
        **QWEN3_VL_VISUAL_CONFIG,
    )
)


register_model_profile(
    ModelProfile(
        model_type="qwen3_vl",
        model_family="qwen3_vl",  # Vision language model, belongs to qwen3_vl family
        vl_patch_method=patch_method_for_qwen3_vl,
        **QWEN3_VL_VISUAL_CONFIG,
    )
)
