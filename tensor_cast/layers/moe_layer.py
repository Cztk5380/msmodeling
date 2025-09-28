from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from .. import ops  # noqa: F401
from ..model_config import MoEConfig
from ..parallel_group import _DEFAULT_PG, ParallelGroup
from .utils import ModelWrapperBase


def assign_experts(num_experts, world_size, rank):
    num_experts_per_device = num_experts // world_size
    num_experts_rest = num_experts % world_size
    if rank < num_experts_rest:
        start = rank * (num_experts_per_device + 1)
        local_num_experts = num_experts_per_device + 1
    else:
        start = (
            num_experts_rest * (num_experts_per_device + 1)
            + (rank - num_experts_rest) * num_experts_per_device
        )
        local_num_experts = num_experts_per_device

    return start, local_num_experts


class FusedMoEBase(torch.nn.Module, ABC):
    def __init__(
        self,
        moe_config: MoEConfig,
        experts: torch.nn.ModuleList,
        shared_experts: Optional[torch.nn.Module],
        shared_experts_gate: Optional[torch.nn.Module],
    ):
        super().__init__()
        self.moe_config = moe_config
        self.experts = experts
        self.shared_experts = shared_experts
        self.shared_experts_gate = shared_experts_gate

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
        module: torch.nn.Module,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.gate = self.get_attr(module, "gate", None)
        self.top_k = self.get_attr(module, "top_k", None)
        self.norm_topk_prob = self.get_attr(module, "norm_topk_prob", None)
        fused_moe_cls = (
            moe_config.fused_moe_cls if moe_config.fused_moe_cls else FusedMoETensorCast
        )
        self.fused_moe = fused_moe_cls(
            self.moe_config,
            self.get_attr(module, "experts", None),
            self.get_attr(module, "shared_experts", None),
            self.get_attr(module, "shared_experts_gate", None),
        )

    def forward(self, hidden_states: torch.Tensor):
        if self.moe_config.gate_returns_raw_logits:
            if self.top_k is None:
                raise ValueError(
                    "top_k must be specified if gate_returns_raw_logits is True"
                )
            router_logits = self.gate(hidden_states)
            topk_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
            topk_weights, topk_indices = torch.topk(topk_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(hidden_states.dtype)
        else:
            topk_indices, topk_weights = self.gate(hidden_states)
            topk_indices = topk_indices.view(
                *hidden_states.shape[:-1], topk_indices.shape[-1]
            )
            topk_weights = topk_weights.view(
                *hidden_states.shape[:-1], topk_weights.shape[-1]
            )

        hidden_states = self.fused_moe(hidden_states, topk_indices, topk_weights)
        return hidden_states

    def get_attr(self, module: torch.nn.Module, name: str, default: Any) -> Any:
        if hasattr(self.moe_config.field_names, name):
            return getattr(module, getattr(self.moe_config.field_names, name), default)
        return default


class ParallelMoELayer(ModelWrapperBase):
    def __init__(
        self,
        module: MoELayer,
        global_dp_group: ParallelGroup,
        global_tp_group: ParallelGroup,
        ep_group: ParallelGroup,
    ):
        super().__init__(module)
        self._inner.fused_moe = FusedMoETensorCast(
            module.moe_config,
            module.fused_moe.experts,
            module.fused_moe.shared_experts,
            module.fused_moe.shared_experts_gate,
            ep_group,
        )
        self.global_dp_group = global_dp_group
        self.global_tp_group = global_tp_group
        self.ep_group = ep_group
        self.transform_dp_group = (
            self.global_dp_group.world_size != self.ep_group.world_size
        )
        if (
            self.transform_dp_group
            and self.ep_group.world_size != self.ep_group.global_world_size
        ):
            raise ValueError(
                f"The scenario where expert_parallel_size {self.ep_group.world_size}"
                f"!= world_size {self.ep_group.global_world_size} is not supported."
            )

    def forward(self, hidden_states: torch.Tensor):
        if self.transform_dp_group:
            origin_shape = hidden_states.shape
            hidden_states = hidden_states.view(-1, *origin_shape[2:])
            hidden_states = self.global_tp_group.slice(hidden_states, dim=0)

        hidden_states = self._inner(hidden_states)

        if self.transform_dp_group:
            hidden_states = self.global_tp_group.all_gather(hidden_states, dim=0)
            hidden_states = hidden_states.view(
                *origin_shape[:2], *hidden_states.shape[1:]
            )

        return hidden_states


class FusedMoETensorCast(FusedMoEBase):
    def __init__(
        self,
        moe_config: MoEConfig,
        experts: torch.nn.ModuleList,
        shared_experts: Optional[torch.nn.Module],
        shared_experts_gate: Optional[torch.nn.Module],
        ep_group: Optional[ParallelGroup] = _DEFAULT_PG,
    ):
        super().__init__(moe_config, experts, shared_experts, shared_experts_gate)
        self.ep_group = ep_group
        self.global_num_experts = len(self.experts)

        expert_idx_start, local_num_experts = assign_experts(
            self.global_num_experts,
            self.ep_group.world_size,
            self.ep_group.rank_in_group,
        )
        self.experts = self.experts[
            expert_idx_start : expert_idx_start + local_num_experts
        ]
        self.local_num_experts = local_num_experts
        self.expert_idx_start = expert_idx_start

    def get_split_sizes(self, num_tokens: int):
        num_tokens_per_expert = num_tokens // self.global_num_experts
        num_tokens_rest = num_tokens % self.global_num_experts

        input_split_sizes_by_expert = [
            num_tokens_per_expert + (i < num_tokens_rest)
            for i in range(self.global_num_experts)
        ]
        input_split_sizes_by_device = []
        for rank in range(self.ep_group.world_size):
            start, num_experts = assign_experts(
                self.global_num_experts, self.ep_group.world_size, rank
            )
            input_split_sizes_by_device.append(
                sum(input_split_sizes_by_expert[start : start + num_experts])
            )

        output_split_sizes_by_device = [
            input_split_sizes_by_device[self.ep_group.rank_in_group]
        ] * self.ep_group.world_size
        output_split_sizes_by_expert = [
            input_split_sizes_by_expert[
                self.expert_idx_start : self.expert_idx_start + self.local_num_experts
            ]
        ] * self.ep_group.world_size

        return (
            input_split_sizes_by_device,
            output_split_sizes_by_device,
            input_split_sizes_by_expert,
            output_split_sizes_by_expert,
        )

    def rearrange_token_by_expert(
        self,
        x: torch.Tensor,
        split_sizes_by_device: List[int],
        split_sizes_by_expert: List[List[int]],
    ) -> List[torch.Tensor]:
        x = x.split(split_sizes_by_device)
        x = [
            x[rank].split(split_sizes_by_expert[rank])
            for rank in range(self.ep_group.world_size)
        ]
        rearranged_x = []
        for i in range(self.local_num_experts):
            cur_expert_tokens = [x[rank][i] for rank in range(self.ep_group.world_size)]
            rearranged_x.append(torch.cat(cur_expert_tokens, dim=0))
        return rearranged_x

    def rearrange_token_by_device(
        self, x: List[torch.Tensor], split_sizes_by_expert: List[List[int]]
    ) -> torch.Tensor:
        for i in range(self.local_num_experts):
            split_sizes = [
                split_sizes_by_expert[rank][i]
                for rank in range(self.ep_group.world_size)
            ]
            x[i] = x[i].split(split_sizes)

        rearranged_x = []
        for rank in range(self.ep_group.world_size):
            for i in range(self.local_num_experts):
                rearranged_x.append(x[i][rank])
        return torch.cat(rearranged_x, dim=0)

    def dispatch_tokens(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        input_split_sizes_by_device: List[int],
        output_split_sizes_by_device: List[int],
        output_split_sizes_by_expert: List[List[int]],
    ) -> List[torch.Tensor]:
        x = torch.ops.tensor_cast.permute_tokens(x, topk_indices)
        dispatched_x = self.ep_group.all_to_all(
            x, output_split_sizes_by_device, input_split_sizes_by_device
        )
        dispatched_x = self.rearrange_token_by_expert(
            dispatched_x, output_split_sizes_by_device, output_split_sizes_by_expert
        )
        return dispatched_x

    def combine_tokens(
        self,
        x: List[torch.Tensor],
        topk_indices: torch.Tensor,
        input_split_sizes_by_device: List[int],
        output_split_sizes_by_device: List[int],
        output_split_sizes_by_expert: List[List[int]],
    ) -> torch.Tensor:
        x = self.rearrange_token_by_device(x, output_split_sizes_by_expert)
        combined_x = self.ep_group.all_to_all(
            x, input_split_sizes_by_device, output_split_sizes_by_device
        )
        combined_x = torch.ops.tensor_cast.unpermute_tokens(combined_x, topk_indices)
        return combined_x

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,  # [bsz, seq, topk]
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: support quantization
        num_tokens = topk_indices.numel()
        split_sizes = self.get_split_sizes(num_tokens)

        dispatched_hidden_states = self.dispatch_tokens(
            hidden_states, topk_indices, split_sizes[0], split_sizes[1], split_sizes[3]
        )

        assert len(dispatched_hidden_states) == len(self.experts)

        experts_hidden_states = []
        for expert_idx in range(len(self.experts)):
            expert_output = self.experts[expert_idx](
                dispatched_hidden_states[expert_idx]
            )
            experts_hidden_states.append(expert_output)

        combined_hidden_states = self.combine_tokens(
            experts_hidden_states,
            topk_indices,
            split_sizes[0],
            split_sizes[1],
            split_sizes[3],
        )
        final_hidden_states = (combined_hidden_states * topk_weights.unsqueeze(-1)).sum(
            dim=-2
        )

        if self.shared_experts:
            shared_expert_output = self.shared_experts(hidden_states)
            if self.shared_experts_gate:
                shared_expert_output = (
                    torch.nn.functional.sigmoid(self.shared_experts_gate(hidden_states))
                    * shared_expert_output
                )
            final_hidden_states = final_hidden_states + shared_expert_output

        return final_hidden_states.to(hidden_states.dtype)
