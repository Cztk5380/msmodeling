from typing import Optional

import torch


class performance_model:
    class empirical:
        runtime_device_override: Optional[torch.device] = None
        warmup_runs = 1
        benchmark_runs = 10


class compilation:
    enable_freezing = True

    class passes:
        enable_life_combine_quant = True
        enable_merge_linear = True

    class fusion_patterns:
        enable_rms_norm = True
        enable_rms_norm_quant = enable_rms_norm
        enable_add_rms_norm = enable_rms_norm
        enable_rope = True

    class debug:
        graph_log_url = None
