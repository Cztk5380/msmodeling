class compilation:
    class passes:
        enable_life_combine_quant = True

    class fusion_patterns:
        enable_rms_norm = True
        enable_rms_norm_quant = enable_rms_norm
        enable_add_rms_norm = enable_rms_norm
        enable_rope = True

    class debug:
        graph_log_url = None
