from typing import Tuple

import torch.nn as nn

from .parser import get_candidate_layers


def trim(
    model: nn.Module,
    layer_idx: int = -1,
    freeze: bool = False,
    input_size: Tuple = (128,),
) -> nn.Module:
    """Trim an arbitary Keras model to a Pytorch embedding model

    :param model: an arbitary DNN model in Pytorch
    :param layer_idx: the index of the bottleneck layer for embedding output.
    :param freeze: if set, the remaining layers of the model will be freezed.
    :param input_size: the input shape to the DNN model.
    """
    candidate_layers = get_candidate_layers(model, input_size=input_size)

    indx = {l['layer_idx'] for l in candidate_layers}
    if layer_idx not in indx:
        raise IndexError(f'Layer index {layer_idx} is not one of {indx}.')

    for idx, module in enumerate(model[layer_idx + 1 :]):
        model[layer_idx + idx + 1] = nn.Identity()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model
