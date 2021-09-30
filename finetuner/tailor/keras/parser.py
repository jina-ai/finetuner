from tensorflow.keras import Model

from ..helper import CandidateLayerInfo


def _get_candidate_layers(model: Model) -> CandidateLayerInfo:
    """Get all dense layers that can be used as embedding layer from the given model. """
    results = []
    for idx, layer in enumerate(model.layers):
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = 'multiple'
        except RuntimeError:  # output_shape unknown in Eager mode.
            output_shape = '?'

        if len(output_shape) != 2:
            continue
        else:
            if not layer.built and not getattr(layer, '_is_graph_network', False):
                # If a subclassed model has a layer that is not called in Model.call, the
                # layer will not be built and we cannot call layer.count_params().
                params = 0
            else:
                params = layer.count_params()

            results.append(
                {
                    'name': layer.name,
                    'cls_name': layer.__class__.__name__,
                    'output_features': output_shape[-1],
                    'params': params,
                    'layer_idx': idx,
                    'module_name': layer.name,  # duplicate as `name` to make different backends consistent
                }
            )
    return results


def _trim(model: Model, layer_idx: int = -1) -> Model:
    """Trim an arbitrary Keras model to a Keras embedding model

    :param model: an arbitary DNN model in Keras
    :param layer_idx: the index of the bottleneck layer for embedding output.

    ..note::
        The argument `layer_idx` means that all layers before (not include) the index will be
        preserved.
    """
    candidate_layers = _get_candidate_layers(model)
    indx = {l['layer_idx'] for l in candidate_layers if l['layer_idx'] != 0}
    if layer_idx not in indx:
        raise IndexError(f'Layer index {layer_idx} is not one of {indx}.')
    return Model(model.input, model.layers[layer_idx - 1].output)


def _freeze(model: Model) -> Model:
    """Freeze an arbitrary model to make layers not trainable.

    :param model: an arbitrary DNN model in Keras.
    :return: A new model with all layers weights freezed.
    """
    for layer in model.layers:
        layer.trainable = False
    return model
