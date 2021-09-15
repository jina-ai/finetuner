from keras.models import Model

from .parser import get_candidate_layers


def trim(model: Model, layer_idx: int = -1, freeze: bool = False) -> Model:
    """Trim an arbitary Keras model to a Keras embedding model

    :param model: an arbitary DNN model in Keras
    :param layer_idx: the index of the bottleneck layer for embedding output.
    :param freeze: if set, the remaining layers of the model will be freezed.
    """
    candidate_layers = get_candidate_layers(model)

    indx = {l['layer_idx'] for l in candidate_layers}
    if layer_idx not in indx:
        raise IndexError(f'Layer index {layer_idx} is not one of {indx}.')
    model = Model(model.input, model.layers[layer_idx].output)

    if freeze:
        for layer in model.layers:
            layer.trainable = False
    return model
