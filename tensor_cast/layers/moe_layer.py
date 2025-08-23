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
        shared_experts: Optional[torch.nn.Module],
    ):
        super().__init__()
        self.moe_config = moe_config
        self.experts = experts
        self.shared_experts = shared_experts

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
        self.fused_moe = fused_moe_cls(self.moe_config, experts, shared_experts)

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
        shared_experts: Optional[torch.nn.Module],
    ):
        super().__init__(moe_config, experts, shared_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: support quantization
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        dispatched_hidden_states, dispatched_indices, dispatched_weights = (
            torch.ops.tensor_cast.dispatch_tokens(
                hidden_states, topk_indices, topk_weights, len(self.experts)
            )
        )
        assert (
            len(dispatched_hidden_states)
            == len(dispatched_weights)
            == len(self.experts)
        )
        for expert_idx in range(len(self.experts)):
            expert_output = self.experts[expert_idx](
                dispatched_hidden_states[expert_idx]
            )
            weighted_output = expert_output * dispatched_weights[expert_idx].unsqueeze(
                -1
            )
            final_hidden_states.index_add_(
                0, dispatched_indices[expert_idx], weighted_output
            )
        if self.shared_experts:
            final_hidden_states = final_hidden_states + self.shared_experts(
                hidden_states
            )
        return final_hidden_states.to(hidden_states.dtype)
