from typing import Tuple
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from ..base import BaseTailor
from ...helper import AnyDNN, is_list_int, EmbeddingLayerInfo


class PytorchTailor(BaseTailor):
    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_dtype: str = 'float32',
        *args,
        **kwargs,
    ):
        """Tailor class for PyTorch DNN models

        :param input_size: a sequence of integers defining the shape of the input tensor. Note, batch size is *not* part
            of ``input_size``.
        :param input_dtype: the data type of the input tensor.
        """
        super().__init__(*args, **kwargs)

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        self._input_size = input_size
        self._input_dtype = input_dtype

    @property
    def embedding_layers(self) -> EmbeddingLayerInfo:
        """Get all dense layers that can be used as embedding layer from the :py:attr:`.model`.

        :return: layers info as :class:`list` of :class:`dict`.
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

                m_key = f'{class_name.lower()}_{module_idx + 1}'
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
                or not is_list_int(output_shape)
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
        if not self._embedding_layer_name:
            module_name = self.embedding_layers[-1]['module_name']
        else:
            _embed_layers = {l['name']: l for l in self.embedding_layers}
            try:
                module_name = _embed_layers[self._embedding_layer_name]['module_name']
            except KeyError:
                raise KeyError(
                    f'The emebdding layer name {self._embedding_layer_name} does not exist.'
                )

        _is_after_embedding_layer = False
        for name, module in self._model.named_modules():
            if name == module_name:
                _is_after_embedding_layer = True
            if _is_after_embedding_layer:
                if (
                    '.' in name
                ):  # Note: in torchvision, nested layer names are named with '.' e.g. classifier.0
                    nested_module, layer = name.split('.')
                    setattr(getattr(self._model, nested_module), layer, nn.Identity())
                else:
                    setattr(self._model, name, nn.Identity())

    def _freeze_weights(self):
        for param in self._model.parameters():
            param.requires_grad = False

    def __call__(self, *args, **kwargs):
        self._trim()
        if self._freeze:
            self._freeze_weights()
