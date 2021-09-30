from collections import OrderedDict
from typing import Tuple

import numpy as np
import paddle
from paddle import nn, Tensor

from ..helper import _is_list_int, CandidateLayerInfo


def _get_candidate_layers(
    model: nn.Layer, input_size: Tuple[int, ...], input_dtype: str = 'float32'
) -> CandidateLayerInfo:
    """Get all dense layers that can be used as embedding layer from the given model. """
    dtypes = [input_dtype] * len(input_size)
    depth = len(list(model.sublayers()))
    for name, layer in model.named_sublayers():
        layer.name = name

    def _get_output_shape(output):
        if isinstance(output, (list, tuple)):
            output_shape = [_get_output_shape(o) for o in output]
        else:
            output_shape = list(output.shape)
        return output_shape

    def register_hook(layer):
        def hook(layer, input, output):

            class_name = str(layer.__class__).split(".")[-1].split("'")[0]

            try:
                layer_idx = int(layer._full_name.split('_')[-1])
            except:
                layer_idx = len(summary)

            m_key = f'{class_name}-{layer_idx + 1}'

            summary[m_key] = OrderedDict()
            summary[m_key]['cls_name'] = layer.__class__.__name__
            summary[m_key]['name'] = layer._full_name
            summary[m_key]['output_shape'] = _get_output_shape(output)
            summary[m_key]['module_name'] = layer.name

            params = 0
            if paddle.in_dynamic_mode():
                layer_state_dict = layer._parameters
            else:
                layer_state_dict = layer.state_dict()

            for k, v in layer_state_dict.items():
                params += np.prod(v.shape)

            summary[m_key]['nb_params'] = params

        if (
            not isinstance(layer, nn.Sequential)
            and not isinstance(layer, nn.LayerList)
            and (not (layer == model) or depth < 1)
        ):

            hooks.append(layer.register_forward_post_hook(hook))
        # For rnn, gru and lstm layer
        elif hasattr(layer, 'could_use_cudnn') and layer.could_use_cudnn:
            hooks.append(layer.register_forward_post_hook(hook))

    if isinstance(input_size, tuple):
        input_size = [input_size]

    x = [
        paddle.cast(paddle.rand([2, *in_size]), dtype)
        for in_size, dtype in zip(input_size, dtypes)
    ]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    results = []
    for idx, layer in enumerate(summary):
        output_shape = summary[layer]['output_shape']
        if not output_shape or len(output_shape) != 2 or not _is_list_int(output_shape):
            continue

        results.append(
            {
                'name': summary[layer]['name'],
                'cls_name': summary[layer]['cls_name'],
                'output_features': output_shape[-1],
                'params': summary[layer]['nb_params'],
                'layer_idx': idx,
                'module_name': summary[layer]['module_name'],
            }
        )

    return results


def _trim(
    model: nn.Layer,
    layer_idx: int = -1,
    input_size: Tuple = (128,),
    input_dtype: str = 'float32',
) -> nn.Layer:
    """
    Trim an arbitrary model to a Paddle embedding model.

    :param model: an arbitrary DNN model in Paddle.
    :param layer_idx: the index of the bottleneck layer for embedding output.
    :param input_size: the input shape to the DNN model.
    :param input_dtype: data type of the input.
    :return: The trimmed model where all layers greater than `layer_idx` been replaced with :class:`nn.Identity`.

    ..note::
        The trim method can only trim model of depth 2, e.g. 2 level of nested nn.Module.
    ..note::
        The argument `layer_idx` means that all layers before (not include) the index will be
        preserved.
    """
    candidate_layers = _get_candidate_layers(
        model, input_size=input_size, input_dtype=input_dtype
    )
    indx = {l['layer_idx'] for l in candidate_layers if l['layer_idx'] != 0}
    if layer_idx not in indx:
        raise IndexError(f'Layer index {layer_idx} is not one of {indx}.')

    module_name = None
    for candidate_layer in candidate_layers:
        if candidate_layer['layer_idx'] == layer_idx:
            module_name = candidate_layer['module_name']
            break

    flag = False
    for name, module in model.named_sublayers():
        if name == module_name:
            flag = True
        if flag:
            if (
                '.' in name
            ):  # Note: in paddle.vision, nested layer names are named with '.' e.g. classifier.0
                nested_module, layer = name.split('.')
                setattr(getattr(model, nested_module), layer, _Identity())
            else:
                setattr(model, name, _Identity())
    return model


def _freeze(model: nn.Layer) -> nn.Layer:
    """Freeze an arbitrary model to make layers not trainable.
    :param model: an arbitrary DNN model in Keras.
    :return: A new model with all layers weights freezed.
    """
    for param in model.parameters():
        param.trainable = False
    return model


class _Identity(nn.Layer):
    """A placeholder identity operator that is argument-insensitive."""

    def forward(self, input_: Tensor) -> Tensor:
        return input_
