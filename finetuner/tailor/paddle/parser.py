from collections import OrderedDict
from typing import Tuple

import numpy as np
import paddle
from paddle import nn

from finetuner.tailor.helper import is_list_int


def get_candidate_layers(
    model: nn.Layer, input_size: Tuple[int, ...], input_dtype: str = 'float32'
):
    dtypes = [input_dtype] * len(input_size)

    depth = len(list(model.sublayers()))

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
        if not output_shape or len(output_shape) != 2 or not is_list_int(output_shape):
            continue

        results.append(
            {
                'name': summary[layer]['name'],
                'cls_name': summary[layer]['cls_name'],
                'output_features': output_shape[-1],
                'params': summary[layer]['nb_params'],
                'layer_idx': idx,
                'module_name': layer.name,
            }
        )

    return results
