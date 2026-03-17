from ..custom_model_registry import ModelProfile, register_model_profile

register_model_profile(
    ModelProfile(
        model_type="mimo_v2_flash",
        moe_module_name="MiMoV2MoE",
        moe_num_experts_key="n_routed_experts",
        mtp_block_module_name="MiMoV2DecoderLayer",
    )
)
