import torch

_executed = set()

def run_once(key, fn, *args, **kargs):
    if (key, fn) not in _executed:
        _executed.add((key, fn))
        fn(*args, **kargs)


def is_view_op(op):
    return op.is_view or op == torch.ops.aten._unsafe_view.default