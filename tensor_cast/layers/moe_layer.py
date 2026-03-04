import copy
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
        num_local_experts = num_experts_per_device + 1
    else:
        start = (
            num_experts_rest * (num_experts_per_device + 1)
            + (rank - num_experts_rest) * num_experts_per_device
        )
        num_local_experts = num_experts_per_device

    return start, num_local_experts


class FusedMoEBase(torch.nn.Module, ABC):
    def __init__(
        self,
        moe_config: MoEConfig,
        experts: torch.nn.ModuleList,
        shared_experts: Optional[torch.nn.Module],
        shared_experts_gate: Optional[torch.nn.Module],
        top_k: Optional[int],
    ):
        super().__init__()
        self.moe_config = moe_config
        self.experts = experts
        self.shared_experts = shared_experts
        self.shared_experts_gate = shared_experts_gate
        self.top_k = top_k

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
        self.top_k = self.get_attr(
            module, "top_k", self.get_attr(self.gate, "top_k", None)
        )
        self.norm_topk_prob = self.get_attr(
            module, "norm_topk_prob", self.get_attr(self.gate, "norm_topk_prob", None)
        )

        fused_moe_cls = (
            moe_config.fused_moe_cls if moe_config.fused_moe_cls else FusedMoETensorCast
        )
        self.fused_moe = fused_moe_cls(
            self.moe_config,
            self.get_attr(module, "experts", None),
            self.get_attr(module, "shared_experts", None),
            self.get_attr(module, "shared_experts_gate", None),
            self.top_k,
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
            gate_output = self.gate(hidden_states)
            # Handle both 2-tuple and 3-tuple returns from gate
            # Some gates return (topk_indices, topk_weights)
            # Others return (topk_indices, topk_weights, router_logits)
            if isinstance(gate_output, tuple) and len(gate_output) >= 2:
                topk_indices = gate_output[0]
                topk_weights = gate_output[1]
                # Ignore router_logits if present (gate_output[2])
            else:
                raise ValueError(
                    f"Expected gate to return tuple with at least 2 elements, got {type(gate_output)}"
                )
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
        num_external_shared_experts: int,
        num_redundant_experts: int,
    ):
        super().__init__(module)
        self.ep_group = ep_group
        self.has_ep = self.ep_group.world_size > 1
        self.num_external_shared_experts = num_external_shared_experts
        self.num_redundant_experts = num_redundant_experts

        moe_config = module.moe_config
        experts = module.fused_moe.experts
        shared_experts = module.fused_moe.shared_experts
        shared_experts_gate = module.fused_moe.shared_experts_gate
        num_routing_experts = len(experts)

        if moe_config.enable_external_shared_experts:
            assert shared_experts is not None
            if ep_group.rank_in_group < num_external_shared_experts:
                experts = None
            else:
                shared_experts, shared_experts_gate = None, None

        if experts is not None and num_redundant_experts > 0:
            for _ in range(num_redundant_experts):
                experts.append(copy.deepcopy(experts[0]))

        self._inner.fused_moe = FusedMoETensorCast(
            moe_config,
            experts,
            shared_experts,
            shared_experts_gate,
            self.top_k,
            self.ep_group,
            num_external_shared_experts=num_external_shared_experts,
            num_global_experts=num_routing_experts + num_redundant_experts,
        )

        self.global_dp_group = global_dp_group
        self.global_tp_group = global_tp_group
        if self.has_ep:
            self.transform_dp_group = (
                self.global_dp_group.world_size != self.ep_group.world_size
            )
        else:
            self.transform_dp_group = self.global_dp_group.world_size != 1

    def forward(self, hidden_states: torch.Tensor):
        if self.transform_dp_group:
            origin_shape = hidden_states.shape
            if len(origin_shape) == 3:
                hidden_states = hidden_states.view(-1, *origin_shape[2:])
            if self.has_ep:
                hidden_states = self.global_tp_group.slice(hidden_states, dim=0)
            else:
                hidden_states = self.global_dp_group.all_gather(hidden_states, dim=0)

        hidden_states = self._inner(hidden_states)

        if self.transform_dp_group:
            if self.has_ep:
                hidden_states = self.global_tp_group.all_gather(hidden_states, dim=0)
            else:
                hidden_states = self.global_dp_group.slice(hidden_states, dim=0)
            if len(origin_shape) == 3:
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
        top_k: Optional[int],
        ep_group: Optional[ParallelGroup] = _DEFAULT_PG,
        num_external_shared_experts: int = 0,
        num_global_experts: Optional[int] = None,
    ):
        super().__init__(
            moe_config, experts, shared_experts, shared_experts_gate, top_k
        )
        self.ep_group = ep_group
        self.num_global_experts = (
            num_global_experts if num_global_experts else len(self.experts)
        )
        self.num_external_shared_experts = num_external_shared_experts

        if self.experts is not None:
            expert_idx_start, num_local_experts = assign_experts(
                self.num_global_experts,
                self.ep_group.world_size - num_external_shared_experts,
                self.ep_group.rank_in_group - num_external_shared_experts,
            )
            self.experts = self.experts[
                expert_idx_start : expert_idx_start + num_local_experts
            ]
            self.num_local_experts = num_local_experts
            self.expert_idx_start = expert_idx_start

    def get_split_sizes(self, num_tokens: int, top_k: int):
        num_tokens_per_expert = num_tokens // self.num_global_experts
        num_tokens_rest = num_tokens % self.num_global_experts

        input_split_sizes_by_expert = [
            num_tokens_per_expert + (i < num_tokens_rest)
            for i in range(self.num_global_experts)
        ]

        input_split_sizes_by_device = []
        if self.num_external_shared_experts > 0:
            num_tokens_per_device = (
                num_tokens // top_k // self.num_external_shared_experts
            )
            num_tokens_rest = num_tokens // top_k % self.num_external_shared_experts
            for rank in range(self.num_external_shared_experts):
                input_split_sizes_by_device.append(
                    num_tokens_per_device + (rank < num_tokens_rest)
                )

        for rank in range(self.num_external_shared_experts, self.ep_group.world_size):
            start, num_experts = assign_experts(
                self.num_global_experts,
                self.ep_group.world_size - self.num_external_shared_experts,
                rank - self.num_external_shared_experts,
            )
            input_split_sizes_by_device.append(
                sum(input_split_sizes_by_expert[start : start + num_experts])
            )

        output_split_sizes_by_device = [
            input_split_sizes_by_device[self.ep_group.rank_in_group]
        ] * self.ep_group.world_size
        if self.ep_group.rank_in_group >= self.num_external_shared_experts:
            output_split_sizes_by_expert = [
                input_split_sizes_by_expert[
                    self.expert_idx_start : self.expert_idx_start
                    + self.num_local_experts
                ]
            ] * self.ep_group.world_size
        else:
            output_split_sizes_by_expert = None

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
        split_sizes_by_expert: Optional[List[List[int]]],
    ) -> List[torch.Tensor]:
        if self.ep_group.rank_in_group < self.num_external_shared_experts:
            return [x]

        x = x.split(split_sizes_by_device)
        x = [
            x[rank].split(split_sizes_by_expert[rank])
            for rank in range(self.ep_group.world_size)
        ]
        rearranged_x = []
        for i in range(self.num_local_experts):
            cur_expert_tokens = [x[rank][i] for rank in range(self.ep_group.world_size)]
            rearranged_x.append(torch.cat(cur_expert_tokens, dim=0))
        # TODO(jgong5): We deliberately concat and then split rearranged_x here to form
        # a split pattern in the graph so that the sink_split_pass can recognize the pattern
        # and is able to do horizontal fusion for experts later on. This is a workaround
        # for now. A better way is to directly recognize the split+split+concat pattern
        # in the graph without the need of hacking the python script here.
        rearranged_x_tensor = torch.cat(rearranged_x, dim=0)
        return list(
            torch.split_with_sizes(
                rearranged_x_tensor, [t.shape[0] for t in rearranged_x], dim=0
            )
        )

    def rearrange_token_by_device(
        self, x: List[torch.Tensor], split_sizes_by_expert: Optional[List[List[int]]]
    ) -> torch.Tensor:
        if self.ep_group.rank_in_group < self.num_external_shared_experts:
            return x[0]

        for i in range(self.num_local_experts):
            split_sizes = [
                split_sizes_by_expert[rank][i]
                for rank in range(self.ep_group.world_size)
            ]
            x[i] = x[i].split(split_sizes)

        rearranged_x = []
        for rank in range(self.ep_group.world_size):
            for i in range(self.num_local_experts):
                rearranged_x.append(x[i][rank])
        return torch.cat(rearranged_x, dim=0)

    def dispatch_tokens(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        input_split_sizes_by_device: List[int],
        output_split_sizes_by_device: List[int],
        output_split_sizes_by_expert: Optional[List[List[int]]],
    ) -> List[torch.Tensor]:
        x = torch.ops.tensor_cast.permute_tokens(x, expert_indices)
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
        expert_indices: torch.Tensor,
        input_split_sizes_by_device: List[int],
        output_split_sizes_by_device: List[int],
        output_split_sizes_by_expert: Optional[List[List[int]]],
    ) -> torch.Tensor:
        x = self.rearrange_token_by_device(x, output_split_sizes_by_expert)
        combined_x = self.ep_group.all_to_all(
            x, input_split_sizes_by_device, output_split_sizes_by_device
        )
        combined_x = torch.ops.tensor_cast.unpermute_tokens(combined_x, expert_indices)
        return combined_x

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,  # [bsz, seq, topk]
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = topk_indices.numel()
        split_sizes = self.get_split_sizes(num_tokens, self.top_k)

        if self.num_external_shared_experts > 0:
            expert_indices = torch.cat(
                [
                    topk_indices,
                    torch.empty(
                        *topk_indices.shape[:-1], 1, device=hidden_states.device
                    ),
                ],
                dim=-1,
            )
            expert_weights = torch.cat(
                [
                    topk_weights,
                    torch.ones(
                        *topk_weights.shape[:-1], 1, device=hidden_states.device
                    ),
                ],
                dim=-1,
            )
        else:
            expert_indices = topk_indices
            expert_weights = topk_weights

        dispatched_hidden_states = self.dispatch_tokens(
            hidden_states,
            expert_indices,
            split_sizes[0],
            split_sizes[1],
            split_sizes[3],
        )

        experts_hidden_states = []
        if self.ep_group.rank_in_group < self.num_external_shared_experts:
            assert len(dispatched_hidden_states) == 1
            shared_expert_output = self.shared_experts(dispatched_hidden_states[0])
            if self.shared_experts_gate:
                shared_expert_output = (
                    torch.nn.functional.sigmoid(
                        self.shared_experts_gate(dispatched_hidden_states[0])
                    )
                    * shared_expert_output
                )
            experts_hidden_states.append(shared_expert_output)
        else:
            assert len(dispatched_hidden_states) == len(self.experts)
            for expert_idx in range(len(self.experts)):
                expert_output = self.experts[expert_idx](
                    dispatched_hidden_states[expert_idx]
                )
                experts_hidden_states.append(expert_output)

        combined_hidden_states = self.combine_tokens(
            experts_hidden_states,
            expert_indices,
            split_sizes[0],
            split_sizes[1],
            split_sizes[3],
        )
        final_hidden_states = (
            combined_hidden_states * expert_weights.unsqueeze(-1)
        ).sum(dim=-2)

        if self.shared_experts and self.num_external_shared_experts == 0:
            shared_expert_output = self.shared_experts(hidden_states)
            if self.shared_experts_gate:
                shared_expert_output = (
                    torch.nn.functional.sigmoid(self.shared_experts_gate(hidden_states))
                    * shared_expert_output
                )
            final_hidden_states = final_hidden_states + shared_expert_output

        return final_hidden_states.to(hidden_states.dtype)


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
