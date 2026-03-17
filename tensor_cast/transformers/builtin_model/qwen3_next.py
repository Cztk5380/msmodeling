from ..custom_model_registry import ModelProfile, register_model_profile

register_model_profile(
    ModelProfile(
        model_type="qwen3_next",
        moe_module_name="Qwen3NextSparseMoeBlock",
        moe_gate_returns_raw_logits=True,
        moe_field_names_override={
            "shared_experts": "shared_expert",
            "shared_experts_gate": "shared_expert_gate",
        },
    )
)
