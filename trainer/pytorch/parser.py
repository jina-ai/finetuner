from collections import OrderedDict

import torch
from torch import nn
import numpy as np


def get_candidate_layers(model, input_size, dtype=torch.FloatTensor):
    dtypes = [dtype] * len(input_size)

    def _get_output_shape(output):
        if isinstance(output, (list, tuple)):
            output_shape = []
            for o in output:
                if isinstance(
                    o, tuple
                ):  # NOTE: lstm returns output and a tuple of (hidden_state, cell_state).
                    o = o[0]
                output_shape.append([-1] + list(o.size())[1:])
        elif isinstance(output, torch.Tensor):
            output_shape = list(output.size())
        else:
            # NOTE: Transformers has it's own output class.
            # type: transformers.modeling_outputs.ModelOutput
            output_shape = list(output.last_hidden_state.size())
        return output_shape

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f'{class_name}-{module_idx + 1}'
            summary[m_key] = OrderedDict()
            summary[m_key]['cls_name'] = module.__class__.__name__
            summary[m_key]['name'] = m_key
            summary[m_key]['output_shape'] = _get_output_shape(output)

            params = 0
            if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                params += np.prod(list(module.weight.size()))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params += np.prod(list(module.bias.size()))
            summary[m_key]['nb_params'] = params

        if not isinstance(module, nn.Sequential) and not isinstance(
            module, nn.ModuleList
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [
        torch.rand(2, *in_size).type(dtype)
        for in_size, dtype in zip(input_size, dtypes)
    ]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
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
