from typing import Tuple

import torch.nn as nn

from .parser import get_candidate_layers


def trim(
    model: nn.Module,
    layer_idx: int = -1,
    input_size: Tuple = (128,),
    input_dtype: str = 'float32',
) -> nn.Module:
    """Trim an arbitrary model to a Pytorch embedding model.

    :param model: an arbitrary DNN model in Pytorch.
    :param layer_idx: the index of the bottleneck layer for embedding output.
    :param input_size: the input shape to the DNN model.
    :param input_dtype: data type of the input.
    :return: The trimmed model where all layers greater than `layer_idx` been replaced with :class:`nn.Identity`.

    ..note::
        The trim method can only trim model of depth 2, e.g. 2 level of nested nn.Module.
    """
    candidate_layers = get_candidate_layers(
        model, input_size=input_size, input_dtype=input_dtype
    )
    indx = {l['layer_idx'] for l in candidate_layers}
    if layer_idx not in indx:
        raise IndexError(f'Layer index {layer_idx} is not one of {indx}.')

    module_name = None
    for candidate_layer in candidate_layers:
        if candidate_layer['layer_idx'] == layer_idx:
            module_name = candidate_layer['module_name']
            break

    flag = False
    for name, module in model.named_modules():
        if name == module_name:
            flag = True
        if flag:
            if (
                '.' in name
            ):  # Note: in torchvision, nested layer names are named with '.' e.g. classifier.0
                nested_module, layer = name.split('.')
                setattr(getattr(model, nested_module), layer, nn.Identity())
            else:
                setattr(model, name, nn.Identity())

    return model


def freeze(model: nn.Module):
    """Freeze an arbitrary model to make layers not trainable.

    :param model: an arbitrary DNN model in Pytorch.
    :return: A new model with all layers weights freezed.
    """
    for param in model.parameters():
        param.requires_grad = False
    return model
