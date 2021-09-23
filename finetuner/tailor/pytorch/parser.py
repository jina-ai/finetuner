from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torchvision

from finetuner.tailor.helper import is_list_int


def get_candidate_layers(
    model: nn.Module, input_size: Tuple[int, ...], input_dtype: str = 'float32'
):
    dtypes = [getattr(torch, input_dtype)] * len(input_size)
    names = []
    for name, module in model.named_modules():
        if not module._modules:  # module do  not have sub modules.
            names.append(name)

    def _get_output_shape(output):
        if isinstance(output, (list, tuple)):
            output_shape = [_get_output_shape(o) for o in output]
        else:
            output_shape = list(output.shape)
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

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and module.__class__.__name__ not in torchvision.models.__dict__.keys()
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dt) for in_size, dt in zip(input_size, dtypes)]

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
        if not output_shape or len(output_shape) != 2 or not is_list_int(output_shape):
            continue

        results.append(
            {
                'name': summary[layer]['name'],
                'cls_name': summary[layer]['cls_name'],
                'output_features': output_shape[-1],
                'params': summary[layer]['nb_params'],
                'layer_idx': idx,
                'module_name': names[idx],
            }
        )

    return results
