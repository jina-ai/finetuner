import paddle.nn as nn


def freeze(model: nn.Layer) -> nn.Layer:
    """Freeze an arbitrary model to make layers not trainable.
    :param model: an arbitrary DNN model in Keras.
    :return: A new model with all layers weights freezed.
    """
    for param in model.parameters():
        param.trainable = False
    return model
