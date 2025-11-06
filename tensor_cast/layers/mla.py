from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from .. import ops  # noqa: F401
from ..model_config import MlaConfig, MultiheadLatentAttentionQuantConfig
from ..parallel_group import ParallelGroup
from ..utils import exact_division
from .attention import AttentionMetadataBase

from .quant_linear import TensorCastQuantLinear
from .utils import get_partial_sharded


class MultiheadLatentAttentionBase(torch.nn.Module, ABC):
    def __init__(
        self,
        mla_config: MlaConfig,
        mla_module: torch.nn.Module,
        decode_only: bool = False,
    ) -> None:
        super().__init__()
        self.mla_config = mla_config
        self._inner = mla_module
        self.decode_only = decode_only
        self.quant_config: Optional[MultiheadLatentAttentionQuantConfig] = None

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        pass

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.mla_config.field_names, name):
            return getattr(self._inner, getattr(self.mla_config.field_names, name))
        return super().__getattr__(name)

    def quantize_params(self):
        """
        Called during the initialization phase after the inner module is quantized.
        This allows quantization for extra parameters in this module.
        """


# rotary embedding functions copied from DeepSeek-v3 model in Transformers
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.view(-1, cos.shape[-1])
    sin = sin.view(-1, sin.shape[-1])
    q_embed = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.view(-1, cos.shape[-1])
    sin = sin.view(-1, sin.shape[-1])

    s, n, d = q.shape
    q = q.view(s, n, d // 2, 2).transpose(-1, -2).reshape(s, n, d)

    s, d = k.shape
    k = k.view(s, d // 2, 2).transpose(-1, -2).reshape(s, d)

    q_embed = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiheadLatentAttentionTensorCast(MultiheadLatentAttentionBase):
    def __init__(
        self,
        mla_config: MlaConfig,
        mla_module: torch.nn.Module,
        tp_group: ParallelGroup,
        decode_only: bool = False,
    ):
        super().__init__(mla_config, mla_module, decode_only)
        sharded_weight = get_partial_sharded(
            self.kv_b_proj.weight.data,
            tp_group.world_size,
            tp_group.rank_in_group,
            unit_size=self.qk_nope_head_dim + self.v_head_dim,
        )
        self.kv_b_proj_weight_t = sharded_weight.transpose(0, 1)
        kv_b_proj_view = self.kv_b_proj_weight_t.view(
            self.kv_lora_rank,
            exact_division(self.num_heads, tp_group.world_size),
            self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK, W_UV = kv_b_proj_view.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        self.W_UV = W_UV.transpose(0, 1)  # (num_heads, kv_lora_rank, v_head_dim)
        self.W_UK_T = W_UK.permute(
            1, 2, 0
        )  # (num_heads, qk_nope_head_dim, kv_lora_rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        kv_cache_unused: Optional[torch.Tensor] = None,
        attention_meta: Optional[AttentionMetadataBase] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        kv_cache_by_layers = kwargs.pop("kv_cache_by_layers", None)
        kv_cache = kv_cache_by_layers[self.layer_idx] if kv_cache_by_layers else None
        batch_size, seq_length = hidden_states.shape[:-1]
        num_tokens = batch_size * seq_length
        hidden_states = hidden_states.view(num_tokens, -1)  # (num_tokens, hidden_size)
        query_shape = (num_tokens, -1, self.qk_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape)  # (num_tokens, num_heads, qk_head_dim)
        # q_pass: (num_tokens, num_heads, qk_nope_head_dim)
        # q_rot: (num_tokens, num_heads, qk_rope_head_dim)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(
            hidden_states
        )  # (num_tokens, kv_lora_rank + qk_rope_head_dim)
        # k_pass: (num_tokens, kv_lora_rank), k_rot: (num_tokens, qk_rope_head_dim)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_c_normed = self.kv_a_layernorm(k_pass)  # (num_tokens, kv_lora_rank)

        cos, sin = position_embeddings
        # TODO: only two rope algorithms are provided below, needed by DeepSeek but
        #       other models might need more alternatives
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        # TODO: do inplace rotary embedding and save this torch.cat? Perhaps we can do this in graph rewrites?
        q_states = torch.cat((q_pass, q_rot), dim=-1)

        # TODO: support quantization
        if attention_meta is not None:
            torch.ops.tensor_cast.concat_and_cache_mla(
                kv_c_normed, k_rot, kv_cache, attention_meta.slot_mapping
            )

        query_start_loc = attention_meta.query_start_loc if attention_meta else None
        seq_lens = attention_meta.seq_lens if attention_meta else None
        query_lens = attention_meta.query_lens if attention_meta else None
        if self.quant_config is not None:
            quant_config = self.quant_config
            out_dtype = self.quant_config.get_quant_dtype()
            q_states = torch.ops.tensor_cast.quantize(
                q_states,
                quant_config.query_scale,
                quant_config.query_offset,
                out_dtype,
            )
            kv_c_normed = torch.ops.tensor_cast.quantize(
                kv_c_normed,
                quant_config.kv_scale,
                quant_config.kv_offset,
                out_dtype,
            )
            k_rot = torch.ops.tensor_cast.quantize(
                k_rot,
                quant_config.kv_scale,
                quant_config.kv_offset,
                out_dtype,
            )
            # we wrap the attention operation in a custom op since it behaves differently
            # between prefill and decode shapes
            attn_output = torch.ops.tensor_cast.multihead_latent_attention_quant(
                q_states,
                kv_c_normed,
                k_rot,
                kv_cache,
                attention_meta.block_table_tensor
                if attention_meta is not None
                else None,
                query_start_loc,
                seq_lens,
                query_lens,
                self.W_UK_T,
                self.W_UV,
                self.kv_b_proj_weight_t,
                self.v_head_dim,
                query_scale=quant_config.query_scale,
                query_offset=quant_config.query_offset,
                kv_scale=quant_config.kv_scale,
                kv_offset=quant_config.kv_offset,
                kv_projected_scale=quant_config.kv_projected_scale,
                kv_projected_offset=quant_config.kv_projected_offset,
                qk_scale=quant_config.qk_scale,
                qk_offset=quant_config.qk_offset,
                v_scale=quant_config.v_scale,
                v_offset=quant_config.v_offset,
                attention_prob_scale=quant_config.attention_prob_scale,
                attention_prob_offset=quant_config.attention_prob_offset,
                kv_b_proj_scale=self.kv_b_proj_scale,
                kv_b_proj_offset=self.kv_b_proj_offset,
                out_scale=quant_config.out_scale,
                out_offset=quant_config.out_offset,
                out_dtype=hidden_states.dtype,
            )
        else:
            # we wrap the attention operation in a custom op since it behaves differently
            # between prefill and decode shapes
            attn_output = torch.ops.tensor_cast.multihead_latent_attention(
                q_states,
                kv_c_normed,
                k_rot,
                kv_cache,
                attention_meta.block_table_tensor
                if attention_meta is not None
                else None,
                query_start_loc,
                seq_lens,
                query_lens,
                self.W_UK_T,
                self.W_UV,
                self.kv_b_proj_weight_t,
                self.v_head_dim,
            )
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def quantize_params(self):
        assert self.quant_config is not None, (
            "quant_config must be set before quantization"
        )
        out_dtype = self.quant_config.get_quant_dtype()
        kv_b_proj = self.kv_b_proj
        if not isinstance(kv_b_proj, TensorCastQuantLinear):
            raise ValueError("MLA quantization requires kv_b_proj to be quantized")
        self.kv_b_proj_scale = kv_b_proj.weight_scale
        self.kv_b_proj_offset = kv_b_proj.weight_offset
        self.kv_b_proj_weight_t = torch.ops.tensor_cast.quantize(
            self.kv_b_proj_weight_t,
            self.kv_b_proj_scale,
            self.kv_b_proj_offset,
            out_dtype,
        )
        self.W_UK_T = torch.ops.tensor_cast.quantize(
            self.W_UK_T,
            self.kv_b_proj_scale,
            self.kv_b_proj_offset,
            out_dtype,
        )
        self.W_UV = torch.ops.tensor_cast.quantize(
            self.W_UV,
            self.kv_b_proj_scale,
            self.kv_b_proj_offset,
            out_dtype,
        )
