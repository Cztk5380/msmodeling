from ..custom_model_registry import ModelProfile, register_model_profile

register_model_profile(
    ModelProfile(
        model_type="minimax_m2",
        moe_module_name="MiniMaxM2SparseMoeBlock",
        moe_gate_returns_raw_logits=True,
        moe_num_experts_key="num_local_experts",
    )
)
