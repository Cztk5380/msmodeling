import contextlib
import threading

import torch

_meta_autocast_enabled = threading.local()


def _is_meta_autocast_enabled():
    return getattr(_meta_autocast_enabled, "value", False)


def _set_meta_autocast_enabled(enabled):
    _meta_autocast_enabled.value = enabled


_in_support_autocast_for_meta = False


@contextlib.contextmanager
def support_autocast_for_meta():
    """
    PyTorch doesn't support "meta" device for autocast. This prevents us from
    running a PyTorch model that calls autocast, such as those models in the
    Transformers. We have to patch PyTorch to work it around.
    """
    get_autocast_dtype_orig = torch.get_autocast_dtype
    _C_is_autocast_available_orig = torch._C._is_autocast_available
    is_autocast_enabled_orig = torch.is_autocast_enabled
    set_autocast_enabled_orig = torch.set_autocast_enabled

    def get_autocast_dtype(device):
        if device == "meta":
            return torch.half
        else:
            return get_autocast_dtype_orig(device)

    def is_autocast_available(device):
        if device == "meta":
            return True
        else:
            return _C_is_autocast_available_orig(device)

    def is_autocast_enabled(device):
        if device == "meta":
            return _is_meta_autocast_enabled()
        else:
            return is_autocast_enabled_orig(device)

    def set_autocast_enabled(device, enabled):
        if device == "meta":
            _set_meta_autocast_enabled(enabled)
        else:
            set_autocast_enabled_orig(device, enabled)

    global _in_support_autocast_for_meta
    if _in_support_autocast_for_meta:
        yield
        return
    _in_support_autocast_for_meta = True
    torch.get_autocast_dtype = get_autocast_dtype
    torch.is_autocast_enabled = is_autocast_enabled
    torch.set_autocast_enabled = set_autocast_enabled
    torch._C._is_autocast_available = is_autocast_available
    yield
    torch._C._is_autocast_available = _C_is_autocast_available_orig
    torch.set_autocast_enabled = set_autocast_enabled_orig
    torch.is_autocast_enabled = is_autocast_enabled_orig
    torch.get_autocast_dtype = get_autocast_dtype_orig
    _in_support_autocast_for_meta = False


@contextlib.contextmanager
def meta_nonzero_assume_all_nonzero():
    """This allows the torch.nonzero call in the MoE models traceable with meta device"""
    old_flag = torch.fx.experimental._config.meta_nonzero_assume_all_nonzero
    torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = True
    yield
    torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = old_flag


@contextlib.contextmanager
def specialize_float():
    """
    Patch torch._dynamo.config.specialize_float to True, so that the float dtype
    information can be preserved in the graph. We assume floats are specialized
    in our pattern matching passes like RMSNorm for params like eps.
    """
    old_flag = torch._dynamo.config.specialize_float
    torch._dynamo.config.specialize_float = True
    yield
    torch._dynamo.config.specialize_float = old_flag


@contextlib.contextmanager
def patch_fallback_node_due_to_unsupported_type():
    """
    Patch torch._inductor.pattern_matcher.fallback_node_due_to_unsupported_type to always return False,
    so that the pattern matching passes can be applied without being blocked due to meta tensors.
    """
    import torch._inductor.pattern_matcher as pattern_matcher

    if not hasattr(pattern_matcher, "fallback_node_due_to_unsupported_type"):
        yield
        return

    original_func = pattern_matcher.fallback_node_due_to_unsupported_type

    def always_false(*args, **kwargs):
        return False

    pattern_matcher.fallback_node_due_to_unsupported_type = always_false
    yield
    pattern_matcher.fallback_node_due_to_unsupported_type = original_func


@contextlib.contextmanager
def patch_torch():
    """
    Apply all patches to PyTorch.
    """
    with (
        support_autocast_for_meta(),
        meta_nonzero_assume_all_nonzero(),
        specialize_float(),
        patch_fallback_node_due_to_unsupported_type(),
    ):
        yield
