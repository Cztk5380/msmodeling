from . import patterns  # noqa: F401
from .compile_backend import CompilerBackend

_backend = None


def get_backend():
    """
    Get the compilation backend for 'torch.compile'.

    Returns:
        Callable: The compilation backend function.
    """
    global _backend
    if _backend is None:
        _backend = CompilerBackend()
    return _backend
