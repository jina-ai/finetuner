from collections import OrderedDict
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch import nn
from jina.helper import cached_property

from ..base import BaseTailor
from ...helper import is_list_int, EmbeddingLayerInfoType


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
        self._trimmed_output_dim = None

    @cached_property
    def embedding_layers(self) -> EmbeddingLayerInfoType:
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
                or summary[layer]['cls_name'] in self._model.__class__.__name__
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

    @property
    def output_dim(self) -> int:
        """Get the user-defined output dimensionality.

        :return: Output dimension of the attached linear layer

        .. note::
           if user didn't specify :py:attr:`output_dim`, return model's last layer output dim.
        """
        if self._output_dim:
            return self._output_dim
        return self._interpret_output_dim()

    def _interpret_output_dim(self):
        if isinstance(self._input_size, list):
            input_size = list(self._input_size[0])
        else:
            input_size = list(self._input_size)
        input_size.insert(0, 1)  # expand 1 dim to input.
        input_ = torch.rand(tuple(input_size))
        if 'int' in self._input_dtype:
            input_ = input_.type(torch.IntTensor)
        return list(self._model(input_).shape)[1]

    def _trim(self):
        if not self._embedding_layer_name:
            module_name = self.embedding_layers[-1]['module_name']
        else:
            _embed_layers = {l['name']: l for l in self.embedding_layers}
            try:
                module_name = _embed_layers[self._embedding_layer_name]['module_name']
            except KeyError as e:
                raise e

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
        self._trimmed_output_dim = self._interpret_output_dim()

    def _freeze_weights(self):
        for param in self._model.parameters():
            param.requires_grad = False

    def _attach_dense_layer(self):
        """Attach a dense layer to the end of the parsed model.

        .. note::
           The attached dense layer have the same shape as the last layer
           in the parsed model.
           The attached dense layer will ignore the :py:attr:`freeze`, this
           layer always trainable.
        """
        if self._output_dim:
            self._model = nn.Sequential(
                self._model,
                nn.Linear(
                    in_features=self._trimmed_output_dim,
                    out_features=self.output_dim,
                    bias=True,
                ),
            )
