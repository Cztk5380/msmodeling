from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F

from .. import ops  # noqa: F401
from ..model_config import MoEConfig


class FusedMoEBase(torch.nn.Module, ABC):
    def __init__(
        self,
        moe_config: MoEConfig,
        experts: torch.nn.ModuleList,
        hidden_act: str,
        shared_experts: Optional[torch.nn.Module],
    ):
        super().__init__()
        self.moe_config = moe_config
        self.hidden_act = hidden_act

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "FusedMoEBase is an abstract class and should not be instantiated directly"
        )


class MoELayer(torch.nn.Module):
    def __init__(
        self,
        moe_config: MoEConfig,
        gate: torch.nn.Module,
        experts: torch.nn.ModuleList,
        hidden_act: str,
        shared_experts: Optional[torch.nn.Module],
        top_k: Optional[int],
        norm_topk_prob: Optional[bool],
    ):
        super().__init__()
        self.moe_config = moe_config
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = gate
        fused_moe_cls = (
            moe_config.fused_moe_cls if moe_config.fused_moe_cls else FusedMoETensorCast
        )
        self.fused_moe = fused_moe_cls(
            self.moe_config, experts, hidden_act, shared_experts
        )

    def forward(self, hidden_states: torch.Tensor):
        if self.moe_config.gate_returns_raw_logits:
            if self.top_k is None:
                raise ValueError(
                    "top_k must be specified if gate_returns_raw_logits is True"
                )
            router_logits = self.gate(hidden_states)
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            topk_weights, topk_indices = torch.topk(topk_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(hidden_states.dtype)
        else:
            topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = self.fused_moe(hidden_states, topk_indices, topk_weights)
        return hidden_states


class FusedMoETensorCast(FusedMoEBase):
    def __init__(
        self,
        moe_config: MoEConfig,
        experts: torch.nn.ModuleList,
        hidden_act: str,
        shared_experts: Optional[torch.nn.Module],
    ):
        super().__init__(moe_config, experts, hidden_act, shared_experts)
        self.experts_gate = [
            getattr(expert, moe_config.field_names.gate_proj).weight.data.transpose(
                0, 1
            )
            for expert in experts
        ]
        self.experts_up = [
            getattr(expert, moe_config.field_names.up_proj).weight.data.transpose(0, 1)
            for expert in experts
        ]
        self.experts_down = [
            getattr(expert, moe_config.field_names.down_proj).weight.data.transpose(
                0, 1
            )
            for expert in experts
        ]
        for i in range(len(experts)):
            self.register_buffer(f"experts_gate_{i}", self.experts_gate[i])
            self.register_buffer(f"experts_up_{i}", self.experts_up[i])
            self.register_buffer(f"experts_down_{i}", self.experts_down[i])

        if shared_experts:
            shared_experts_gate = getattr(
                shared_experts, moe_config.field_names.gate_proj
            ).weight.data.transpose(0, 1)
            shared_experts_up = getattr(
                shared_experts, moe_config.field_names.up_proj
            ).weight.data.transpose(0, 1)
            shared_experts_down = getattr(
                shared_experts, moe_config.field_names.down_proj
            ).weight.data.transpose(0, 1)
            self.register_buffer("shared_experts_gate", shared_experts_gate)
            self.register_buffer("shared_experts_up", shared_experts_up)
            self.register_buffer("shared_experts_down", shared_experts_down)
        else:
            self.register_buffer("shared_experts_gate", None)
            self.register_buffer("shared_experts_up", None)
            self.register_buffer("shared_experts_down", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: call experts and shared_experts explicitly here by distributing tokens evenly or according to
        #       some configuration, then do GMM fusion in the graph pass
        # TODO: support quantization
        return torch.ops.tensor_cast.fused_moe(
            hidden_states,
            self.experts_gate,
            self.experts_up,
            self.experts_down,
            self.shared_experts_gate,
            self.shared_experts_up,
            self.shared_experts_down,
            topk_indices,
            topk_weights,
            self.hidden_act,
        )
