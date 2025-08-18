import torch
import contextlib
import threading

_meta_autocast_enabled = threading.local()

def _is_meta_autocast_enabled():
    return getattr(_meta_autocast_enabled, "value", False)

def _set_meta_autocast_enabled(enabled):
    _meta_autocast_enabled.value = enabled

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

    torch.get_autocast_dtype = get_autocast_dtype
    torch.is_autocast_enabled = is_autocast_enabled
    torch.set_autocast_enabled = set_autocast_enabled
    torch._C._is_autocast_available = is_autocast_available
    yield
    torch._C._is_autocast_available = _C_is_autocast_available_orig
    torch.set_autocast_enabled = set_autocast_enabled_orig
    torch.is_autocast_enabled = is_autocast_enabled_orig
    torch.get_autocast_dtype = get_autocast_dtype_orig


@contextlib.contextmanager
def meta_nonzero_assume_all_nonzero():
    """This allows the torch.nonzero call in the MoE models traceable with meta device"""
    old_flag = torch.fx.experimental._config.meta_nonzero_assume_all_nonzero
    torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = True
    yield
    torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = old_flag


@contextlib.contextmanager
def patch_torch():
    """
    Apply all patches to PyTorch.
    """
    with support_autocast_for_meta(), meta_nonzero_assume_all_nonzero():
        yield
