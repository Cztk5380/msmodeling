from ..custom_model_registry import ModelProfile, register_model_profile


def patch_method_for_qwen3_5():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

    def _patched_update_linear_attn_mask(self, attention_mask, cache_position):
        """
        Core Conflict:
        During PyTorch's symbolic tracing (e.g., torch.compile or torch.fx),
        input tensors (like cache_position) are Meta Tensors.
        Meta Tensors contain only shape and dtype metadata, no actual data values.

        Error Trigger:
        The original code if cache_position[0] > 0:
        attempts to use the result of a tensor comparison directly in a Python if control flow statement.
        Python's if requires a concrete boolean value (True or False).
        To obtain this, PyTorch implicitly calls .item() to extract the scalar value from the tensor.
        Since Meta Tensors hold no data, calling .item() fails, raising Tensor.item() cannot be called on meta tensors.
        Conclusion:
        In dynamic graph compilation modes,
        you cannot use specific tensor values to dictate Python code execution branches.
        """
        # Currently, this is the only feasible modification. However, the drawback is that
        # it still passes an attention mask to the linear attention mechanism during decoding, where it is unnecessary.
        return attention_mask

    Qwen3_5TextModel._update_linear_attn_mask = _patched_update_linear_attn_mask
    Qwen3_5MoeTextModel._update_linear_attn_mask = _patched_update_linear_attn_mask


register_model_profile(
    ModelProfile(
        model_type="qwen3_5_moe",
        moe_module_name="Qwen3_5MoeSparseMoeBlock",
        moe_gate_returns_raw_logits=True,
        moe_num_experts_key=["text_config", "num_experts"],
        moe_field_names_override={
            "shared_experts": "shared_expert",
            "shared_experts_gate": "shared_expert_gate",
        },
        model_family="qwen3_5",
        patch_method=patch_method_for_qwen3_5,
    )
)


register_model_profile(
    ModelProfile(
        model_type="qwen3_5",
        model_family="qwen3_5",
        patch_method=patch_method_for_qwen3_5,
    )
)
