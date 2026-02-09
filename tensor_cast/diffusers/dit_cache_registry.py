import logging
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

WrappedForwardFactory = Callable[[Callable[..., Any]], Callable[..., Any]]
MakeWrappedForward = Callable[[Any], WrappedForwardFactory]
GetBlocks = Callable[[Any], Sequence[Any]]


@dataclass(frozen=True)
class DiTBlockCacheSpec:
    model_type: str
    get_blocks: GetBlocks
    make_wrapped_forward: MakeWrappedForward


_DIT_BLOCK_CACHE_SPECS: Dict[str, DiTBlockCacheSpec] = {}


def register_dit_block_cache_spec(class_name: str, spec: DiTBlockCacheSpec) -> None:
    if not class_name:
        raise ValueError("'class_name' must be a non-empty string.")
    _DIT_BLOCK_CACHE_SPECS[class_name] = spec


def get_dit_block_cache_spec(class_name: Optional[str]) -> Optional[DiTBlockCacheSpec]:
    if not class_name:
        return None
    return _DIT_BLOCK_CACHE_SPECS.get(class_name)


def wrap_blocks(
    blocks: Iterable[Any], make_wrapped_forward: WrappedForwardFactory
) -> int:
    wrapped = 0
    for block in blocks:
        if hasattr(block, "_tensor_cast_orig_forward"):
            continue
        orig_forward_bound = block.forward
        block._tensor_cast_orig_forward = orig_forward_bound
        block.forward = types.MethodType(
            make_wrapped_forward(orig_forward_bound), block
        )
        wrapped += 1
    return wrapped


def _get_wan_blocks(inner: Any) -> Sequence[Any]:
    if not hasattr(inner, "blocks"):
        logger.warning("WanTransformer3DModel has no attribute 'blocks'.")
        return []
    blocks = list(inner.blocks)
    if not blocks:
        logger.warning("WanTransformer3DModel.blocks is empty.")
    return blocks


def _get_hunyuanvideo_blocks(inner: Any) -> Sequence[Any]:
    if not hasattr(inner, "transformer_blocks"):
        logger.warning(
            "HunyuanVideoTransformer3DModel has no attribute 'transformer_blocks'."
        )
        return []
    if not hasattr(inner, "single_transformer_blocks"):
        logger.warning(
            "HunyuanVideoTransformer3DModel has no attribute 'single_transformer_blocks'."
        )
        return []
    blocks = list(inner.transformer_blocks) + list(inner.single_transformer_blocks)
    if not blocks:
        logger.warning("HunyuanVideoTransformer3DModel transformer blocks are empty.")
    return blocks


def _get_hunyuanvideo15_blocks(inner: Any) -> Sequence[Any]:
    if not hasattr(inner, "transformer_blocks"):
        logger.warning(
            "HunyuanVideo15Transformer3DModel has no attribute 'transformer_blocks'."
        )
        return []
    blocks = list(inner.transformer_blocks)
    if not blocks:
        logger.warning("HunyuanVideo15Transformer3DModel.transformer_blocks is empty.")
    return blocks


def _wan_make_wrapped_forward(agent: Any) -> WrappedForwardFactory:
    def _make_wrapped_forward(
        orig_forward_bound: Callable[..., Any],
    ) -> Callable[..., Any]:
        def _wrapped_forward(
            self_block: Any,
            hidden_states: Any,
            encoder_hidden_states: Any,
            temb: Any,
            rotary_emb: Any,
        ) -> Any:
            return agent.apply(
                orig_forward_bound,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                rotary_emb=rotary_emb,
            )

        return _wrapped_forward

    return _make_wrapped_forward


def _hunyuanvideo_make_wrapped_forward(agent: Any) -> WrappedForwardFactory:
    def _make_wrapped_forward(
        orig_forward_bound: Callable[..., Any],
    ) -> Callable[..., Any]:
        def _wrapped_forward(
            self_block: Any,
            hidden_states: Any,
            encoder_hidden_states: Any,
            temb: Any,
            attention_mask: Any,
            image_rotary_emb: Any,
            token_replace_emb: Any,
            first_frame_num_tokens: Any,
        ) -> Any:
            return agent.apply(
                orig_forward_bound,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                attention_mask=attention_mask,
                freqs_cis=image_rotary_emb,
                token_replace_emb=token_replace_emb,
                first_frame_num_tokens=first_frame_num_tokens,
            )

        return _wrapped_forward

    return _make_wrapped_forward


def _hunyuanvideo15_make_wrapped_forward(agent: Any) -> WrappedForwardFactory:
    def _make_wrapped_forward(
        orig_forward_bound: Callable[..., Any],
    ) -> Callable[..., Any]:
        def _wrapped_forward(
            self_block: Any,
            hidden_states: Any,
            encoder_hidden_states: Any,
            temb: Any,
            encoder_attention_mask: Any,
            image_rotary_emb: Any,
        ) -> Any:
            return agent.apply(
                orig_forward_bound,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                attention_mask=encoder_attention_mask,
                freqs_cis=image_rotary_emb,
            )

        return _wrapped_forward

    return _make_wrapped_forward


register_dit_block_cache_spec(
    "WanTransformer3DModel",
    DiTBlockCacheSpec(
        model_type="Wan",
        get_blocks=_get_wan_blocks,
        make_wrapped_forward=_wan_make_wrapped_forward,
    ),
)
register_dit_block_cache_spec(
    "HunyuanVideoTransformer3DModel",
    DiTBlockCacheSpec(
        model_type="HunyuanVideo",
        get_blocks=_get_hunyuanvideo_blocks,
        make_wrapped_forward=_hunyuanvideo_make_wrapped_forward,
    ),
)
register_dit_block_cache_spec(
    "HunyuanVideo15Transformer3DModel",
    DiTBlockCacheSpec(
        model_type="HunyuanVideo15",
        get_blocks=_get_hunyuanvideo15_blocks,
        make_wrapped_forward=_hunyuanvideo15_make_wrapped_forward,
    ),
)
