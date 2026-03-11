import fnmatch
import importlib
import logging
import os

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import TransformerModel

logger = logging.getLogger(__name__)


_CUSTOM_MODEL_REGISTRY: Dict[str, Callable] = {}
_USER_CUSTOM_MODEL_LOADED = False


def find_matching_key(registry: Dict[str, Any], key: str) -> Optional[str]:
    if not key:
        return None
    for pattern in registry.keys():
        if fnmatch.fnmatchcase(key, pattern) or fnmatch.fnmatch(key, pattern):
            return pattern
    return None


def register_custom_model(model_type: str):
    def decorator(
        fn: Callable[["TransformerModel"], "TransformerModel"],
    ) -> Callable[["TransformerModel"], "TransformerModel"]:
        _CUSTOM_MODEL_REGISTRY[model_type] = fn
        return fn

    return decorator


def get_custom_model(model_type: str) -> Optional[Callable]:
    if not _USER_CUSTOM_MODEL_LOADED:
        import_custom_model_modules()

    match_key = find_matching_key(_CUSTOM_MODEL_REGISTRY, model_type)
    return _CUSTOM_MODEL_REGISTRY.get(match_key) if match_key else None


def import_custom_model_modules():
    global _USER_CUSTOM_MODEL_LOADED
    if _USER_CUSTOM_MODEL_LOADED:
        return

    _PACKAGE_ROOT = os.path.dirname(importlib.util.find_spec("tensor_cast").origin)
    custom_model_path = os.path.join(_PACKAGE_ROOT, "custom_model")
    if not os.path.exists(custom_model_path):
        return
    from tensor_cast import custom_model  # noqa: F401

    _USER_CUSTOM_MODEL_LOADED = True
