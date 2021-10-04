from typing import Tuple
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from ..base import BaseTailor
from ...helper import AnyDNN
from ..helper import CandidateLayerInfo, _is_list_int


class PytorchTailor(BaseTailor):
    def __init__(
        self,
        model: AnyDNN,
        input_size: Tuple[int, ...],
        layer_idx: int = -1,
        freeze: bool = False,
        input_dtype: str = 'float32',
        *args,
        **kwargs,
    ):
        super().__init__(model, layer_idx, freeze, *args, **kwargs)
        self._input_size = input_size
        self._input_dtype = input_dtype

    @property
    def candidate_layers(self) -> CandidateLayerInfo:
        """Get all dense layers that can be used as embedding layer from the given model.

        :return: Candidate layers info as list of dictionary.
        """
        user_model = deepcopy(self._model)
        dtypes = [getattr(torch, self._input_dtype)] * len(self._input_size)

        # assign name to each module from named_module
        for name, module in user_model.named_modules():
            module.name = name

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
                summary[m_key]['module_name'] = module.name

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
        if isinstance(self._input_size, tuple):
            self._input_size = [self._input_size]

        # batch_size of 2 for batchnorm
        x = [
            torch.rand(2, *in_size).type(dt)
            for in_size, dt in zip(self._input_size, dtypes)
        ]

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        user_model.apply(register_hook)

        # make a forward pass
        user_model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()

        results = []
        for idx, layer in enumerate(summary):
            output_shape = summary[layer]['output_shape']
            if (
                not output_shape
                or len(output_shape) != 2
                or not _is_list_int(output_shape)
            ):
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

    def _trim(self):
        """Trim an arbitrary Keras model to a Pytorch embedding model.

        ..note::
            The argument `layer_idx` means that all layers before (not include) the index will be
            preserved.
        """
        indx = {l['layer_idx'] for l in self.candidate_layers if l['layer_idx'] != 0}
        if self._layer_idx not in indx:
            raise IndexError(f'Layer index {self._layer_idx} is not one of {indx}.')

        module_name = None
        for candidate_layer in self.candidate_layers:
            if candidate_layer['layer_idx'] == self._layer_idx:
                module_name = candidate_layer['module_name']
                break

        flag = False
        for name, module in self._model.named_modules():
            if name == module_name:
                flag = True
            if flag:
                if (
                    '.' in name
                ):  # Note: in torchvision, nested layer names are named with '.' e.g. classifier.0
                    nested_module, layer = name.split('.')
                    setattr(getattr(self._model, nested_module), layer, nn.Identity())
                else:
                    setattr(self._model, name, nn.Identity())

    def _freeze_weights(self):
        """Freeze an arbitrary model to make layers not trainable."""
        for param in self._model.parameters():
            param.requires_grad = False

    def __call__(self, *args, **kwargs):
        self._trim()
        if self._freze:
            self._freeze_weights()
