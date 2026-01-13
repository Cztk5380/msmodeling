"""
Attention adapters for models that use non-standard attention interfaces.

This module provides adapters to convert various attention interfaces to the
standard tensorcast interface (query, key, value tensors).
"""

from typing import Optional, Tuple

import torch

from .attention import flash_attention_forward


class BailingMoeV2AttentionAdapter(torch.nn.Module):
    """
    Adapter for BailingMoeV2Attention

    This adapter wraps the original attention module and:
    1. Extracts QKV computation from the original module
    2. Applies necessary transformations (RoPE, QK norm, etc.)
    3. Calls tensorcast's attention ops
    4. Applies output projection

    Args:
        original_attention: The original BailingMoeV2Attention module to wrap
        attention_by_layers: Dictionary mapping layer_idx to AttentionTensorCast instances
    """

    def __init__(self, original_attention: torch.nn.Module, attention_by_layers: dict):
        super().__init__()
        self.original_attention = original_attention
        self.attention_by_layers = attention_by_layers

        # Copy attributes from original attention
        self.layer_idx = original_attention.layer_idx
        self.hidden_size = original_attention.hidden_size
        self.config = original_attention.config
        self.num_heads = original_attention.num_heads
        self.head_dim = original_attention.head_dim
        self.num_key_value_heads = original_attention.num_key_value_heads
        self.num_key_value_groups = original_attention.num_key_value_groups

        # Reference to original module's layers
        self.q_proj = torch.nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=self.config.use_qkv_bias,
        )
        self.k_proj = torch.nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.config.use_qkv_bias,
        )
        self.v_proj = torch.nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.config.use_qkv_bias,
        )
        self.o_proj = torch.nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=self.config.use_bias
        )

        # Optional layers
        if hasattr(original_attention, "query_layernorm"):
            self.query_layernorm = original_attention.query_layernorm
        if hasattr(original_attention, "key_layernorm"):
            self.key_layernorm = original_attention.key_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass that computes QKV and calls tensorcast attention.
        """
        bsz, q_len, _ = hidden_states.size()

        # Compute QKV
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, -1, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, -1, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, -1, self.head_dim)
            .transpose(1, 2)
        )

        # Apply QK norm if configured
        if hasattr(self, "query_layernorm") and self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        # Apply rotary embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Import the apply_rotary_pos_emb from the original module's namespace
            # This ensures we use the correct implementation for this model
            apply_rotary_pos_emb = self._get_apply_rotary_pos_emb()
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Handle KV cache
        if past_key_value is not None:
            cache_kwargs = {}
            if position_embeddings is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Call tensorcast attention
        attn_output, attn_weights = flash_attention_forward(
            self.original_attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            attention_by_layers=self.attention_by_layers,
            **kwargs,
        )

        # Reshape and apply output projection
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    def _get_apply_rotary_pos_emb(self):
        """
        Get the apply_rotary_pos_emb function from the original module's namespace.
        """
        # Get the module where the original attention class is defined
        import sys

        module_name = self.original_attention.__class__.__module__
        if module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, "apply_rotary_pos_emb"):
                return module.apply_rotary_pos_emb

        # Fallback: use a generic implementation
        def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)

            rotary_dim = cos.shape[-1]
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)

            q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
            k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

            q_embed = torch.cat([q_embed, q_pass], dim=-1)
            k_embed = torch.cat([k_embed, k_pass], dim=-1)
            return q_embed, k_embed

        return apply_rotary_pos_emb
