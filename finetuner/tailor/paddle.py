from ..paddle.parser import get_candidate_layers


def trim(model, input_size, layer_idx: int, freeze=True):
    candidate_layers = get_candidate_layers(model, input_size)
    print(candidate_layers)
