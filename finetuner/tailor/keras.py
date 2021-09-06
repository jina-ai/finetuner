from keras.models import Model

from ..keras.parser import get_candidate_layers


def trim(model, layer_idx: int, freeze=True):
    """The method trim the model based on the user input for feature extraction.

    :param model: The base model served as feature extractor.
    :param layer_idx: The layer index to cut from, it should be one of the candidate layers.
    :param freeze: Freeze the weight of the base model without training.
    """
    candidate_layers = get_candidate_layers(model)
    tailorable_indices = [item['layer_idx'] for item in candidate_layers]
    if layer_idx not in tailorable_indices:
        msg = f'Layer index {layer_idx} is not a candidate layer, One of the index in {tailorable_indices} expected.'
        raise IndexError(msg)
    model = Model(model.input, model.layers[layer_idx].output)
    if freeze:
        for layer in model.layers:
            layer.trainable = False
    return model
