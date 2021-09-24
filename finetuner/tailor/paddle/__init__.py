from typing import Tuple

import paddle.nn as nn

from .parser import get_candidate_layers


def trim(
    model: nn.Layer,
    layer_idx: int = -1,
    input_size: Tuple = (128,),
    input_dtype: str = 'float32',
) -> nn.Layer:
    """Trim an arbitrary model to a Paddle embedding model.
    :param model: an arbitrary DNN model in Paddle.
    :param layer_idx: the index of the bottleneck layer for embedding output.
    :param input_size: the input shape to the DNN model.
    :param input_dtype: data type of the input.
    :return: Trimmed model.
    """
    candidate_layers = get_candidate_layers(
        model, input_size=input_size, input_dtype=input_dtype
    )
    pass


def freeze(model: nn.Layer) -> nn.Layer:
    """Freeze an arbitrary model to make layers not trainable.
    :param model: an arbitrary DNN model in Keras.
    :return: A new model with all layers weights freezed.
    """
    for param in model.parameters():
        param.trainable = False
    return model
