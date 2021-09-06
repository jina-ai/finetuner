from ..keras.parser import get_candidate_layers


def tail(model, layer_idx: int, freeze=True):
    """The method cut the model based on the user input for feature extraction.

    :param model: The base model served as feature extractor.
    :param layer_idx: The layer index to cut from, it should be one of the candidate layers.
    :param freeze: Freeze the weight of the base model without training.
    """
    pass
