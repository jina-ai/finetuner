from typing import Tuple
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import paddle
from paddle import nn, Tensor

from ..base import BaseTailor
from ...helper import is_list_int, EmbeddingLayerInfo


class PaddleTailor(BaseTailor):
    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_dtype: str = 'float32',
        *args,
        **kwargs,
    ):
        """Tailor class for Paddle DNN models

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
        dtypes = [self._input_dtype] * len(self._input_size)
        depth = len(list(user_model.sublayers()))
        for name, layer in user_model.named_sublayers():
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
                and (not (layer == user_model) or depth < 1)
            ):

                hooks.append(layer.register_forward_post_hook(hook))
            # For rnn, gru and lstm layer
            elif hasattr(layer, 'could_use_cudnn') and layer.could_use_cudnn:
                hooks.append(layer.register_forward_post_hook(hook))

        x = [
            paddle.cast(paddle.rand([2, *in_size]), dtype)
            for in_size, dtype in zip(self._input_size, dtypes)
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
        for name, module in self._model.named_sublayers():
            if name == module_name:
                _is_after_embedding_layer = True
            if _is_after_embedding_layer:
                if (
                    '.' in name
                ):  # Note: in paddle.vision, nested layer names are named with '.' e.g. classifier.0
                    nested_module, layer = name.split('.')
                    setattr(getattr(self._model, nested_module), layer, _Identity())
                else:
                    setattr(self._model, name, _Identity())

    def _freeze_weights(self):
        """Freeze an arbitrary model to make layers not trainable."""
        for param in self._model.parameters():
            param.trainable = False

    def __call__(self, *args, **kwargs):
        self._trim()
        if self._freeze:
            self._freeze_weights()


class _Identity(nn.Layer):
    """A placeholder identity operator that is argument-insensitive."""

    def forward(self, input_: Tensor) -> Tensor:
        return input_
