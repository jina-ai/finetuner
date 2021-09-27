from keras.models import Model

from .parser import get_candidate_layers


def trim(model: Model, layer_idx: int = -1) -> Model:
    """Trim an arbitary Keras model to a Keras embedding model

    :param model: an arbitary DNN model in Keras
    :param layer_idx: the index of the bottleneck layer for embedding output.

    ..Note::
        The argument `layer_idx` means that all layers before (not include) the index will be
        preserved.
    """
    candidate_layers = get_candidate_layers(model)
    indx = {l['layer_idx'] for l in candidate_layers if l['layer_idx'] != 0}
    if layer_idx not in indx:
        raise IndexError(f'Layer index {layer_idx} is not one of {indx}.')
    return Model(model.input, model.layers[layer_idx - 1].output)


def freeze(model: Model):
    """Freeze an arbitrary model to make layers not trainable.

    :param model: an arbitrary DNN model in Keras.
    :return: A new model with all layers weights freezed.
    """
    for layer in model.layers:
        layer.trainable = False
    return model
