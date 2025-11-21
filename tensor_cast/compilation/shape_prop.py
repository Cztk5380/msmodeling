from torch._dynamo.utils import detect_fake_mode
from torch.fx.passes.shape_prop import ShapeProp


def shape_propagation(fx_graph, inputs):
    ShapeProp(
        gm=fx_graph,
        fake_mode=detect_fake_mode(inputs),
    ).propagate(*tuple(inputs))
