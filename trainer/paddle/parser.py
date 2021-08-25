import numbers
from collections import OrderedDict

import numpy as np
import paddle
from paddle import nn


def get_candidate_layers(model, input_size):
    def _all_is_number(items):
        for item in items:
            if not isinstance(item, numbers.Number):
                return False
        return True

    def _build_dtypes(input_size, dtype):
        if dtype is None:
            dtype = 'float32'

        if isinstance(input_size, (list, tuple)) and _all_is_number(input_size):
            return [dtype]
        else:
            return [_build_dtypes(i, dtype) for i in input_size]

    dtypes = _build_dtypes(input_size, None)

    depth = len(list(model.sublayers()))

    def _get_shape_from_tensor(x):
        if isinstance(x, (paddle.fluid.Variable, paddle.fluid.core.VarBase)):
            return list(x.shape)
        elif isinstance(x, (list, tuple)):
            return [_get_shape_from_tensor(xx) for xx in x]

    def _get_output_shape(output):
        if isinstance(output, (list, tuple)):
            output_shape = [_get_output_shape(o) for o in output]
        else:
            output_shape = list(output.shape)
        return output_shape

    def register_hook(layer):
        def hook(layer, input, output):

            m_key = layer._full_name
            summary[m_key] = OrderedDict()
            summary[m_key]['cls_name'] = layer.__class__.__name__
            summary[m_key]['name'] = layer._full_name
            summary[m_key]["output_shape"] = _get_output_shape(output)

            params = 0
            if paddle.in_dynamic_mode():
                layer_state_dict = layer._parameters
            else:
                layer_state_dict = layer.state_dict()

            for k, v in layer_state_dict.items():
                params += np.prod(v.shape)

            summary[m_key]["nb_params"] = params

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

    def build_input(input_size, dtypes):
        if isinstance(input_size, (list, tuple)) and _all_is_number(input_size):
            if isinstance(dtypes, (list, tuple)):
                dtype = dtypes[0]
            else:
                dtype = dtypes
            return paddle.cast(paddle.rand(list(input_size)), dtype)
        else:
            return [build_input(i, dtype) for i, dtype in zip(input_size, dtypes)]

    x = build_input(input_size, dtypes)

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
        if len(output_shape) != 2:
            continue

        results.append(
            {
                'name': summary[layer]['name'],
                'cls_name': summary[layer]['cls_name'],
                'output_features': output_shape[-1],
                'params': summary[layer]['nb_params'],
                'layer_idx': idx,
            }
        )

    return results
