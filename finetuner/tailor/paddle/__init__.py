import copy
import warnings
from collections import OrderedDict
from typing import Optional

import numpy as np
import paddle
from paddle import nn, Tensor

from ..base import BaseTailor
from ...helper import is_seq_int, LayerInfoType, AnyDNN


class PaddleTailor(BaseTailor):
    """Tailor class for Paddle DNN models.

    .. note::
        To use this class, you need to set ``input_size`` and ``input_dtype`` in :py:meth:`.__init__`
    """

    def summary(self, skip_identity_layer: bool = False) -> LayerInfoType:
        if not self._input_size:
            raise ValueError(
                f'{self.__class__} requires a valid `input_size`, but receiving {self._input_size}'
            )

        user_model = copy.deepcopy(self._model)
        dtypes = [self._input_dtype] * len(self._input_size)
        depth = len(list(user_model.sublayers()))
        for name, layer in user_model.named_sublayers():
            layer.name = name

        def _get_shape(output):
            if isinstance(output, (list, tuple)):
                output_shape = [_get_shape(o) for o in output]
                if len(output) == 1:
                    output_shape = output_shape[0]
            else:
                output_shape = list(output.shape)
            return output_shape

        def register_hook(layer):
            def hook(layer, input, output):

                class_name = str(layer.__class__).split('.')[-1].split("'")[0]

                layer_idx = len(summary)

                m_key = f'{class_name.lower()}_{layer_idx + 1}'

                summary[m_key] = OrderedDict()
                summary[m_key]['cls_name'] = layer.__class__.__name__
                summary[m_key]['name'] = m_key
                summary[m_key]['input_shape'] = _get_shape(input)
                summary[m_key]['output_shape'] = _get_shape(output)
                summary[m_key]['module_name'] = layer.name

                params = 0
                if paddle.in_dynamic_mode():
                    layer_state_dict = layer._parameters
                else:
                    layer_state_dict = layer.state_dict()

                for k, v in layer_state_dict.items():
                    params += np.prod(v.shape)

                summary[m_key]['nb_params'] = params
                summary[m_key]['trainable'] = any(
                    l.trainable for _, l in layer_state_dict.items()
                )

            if (
                not isinstance(layer, nn.Sequential)
                and not isinstance(layer, nn.LayerList)
                and (layer != user_model or depth < 1)
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
            input_shape = summary[layer]['input_shape']
            is_embedding_layer = not (
                not output_shape
                or len(output_shape) != 2
                or not is_seq_int(output_shape)
                or summary[layer]['cls_name'] in self._model.__class__.__name__
            )

            if (
                skip_identity_layer
                and output_shape == input_shape
                and not summary[layer]['nb_params']
            ):
                # not an effective layer, often a wrapper/identity layer
                continue

            results.append(
                {
                    **summary[layer],
                    'output_features': output_shape[-1],
                    'output_shape_display': output_shape[1:],
                    'layer_idx': idx,
                    'is_embedding_layer': is_embedding_layer,
                }
            )
        return results

    def to_embedding_model(
        self,
        layer_name: Optional[str] = None,
        output_dim: Optional[int] = None,
        freeze: bool = False,
    ) -> AnyDNN:
        model = copy.deepcopy(self._model)

        if layer_name:
            _all_embed_layers = {l['name']: l for l in self.embedding_layers}
            try:
                _embed_layer = _all_embed_layers[layer_name]
            except KeyError as e:
                raise KeyError(
                    f'`embedding_layer_name` must be one of {_all_embed_layers.keys()}, given {layer_name}'
                ) from e
        else:
            # when not given, using the last layer
            _embed_layer = self.embedding_layers[-1]

        if freeze:
            for param in model.parameters():
                param.trainable = False

        _relative_idx_to_embedding_layer = None
        _is_dense_layer_added = False
        for name, module in model.named_sublayers():
            if name == _embed_layer['module_name']:
                _relative_idx_to_embedding_layer = 0

                # corner-case
                if not output_dim and not layer_name:
                    for param in module.parameters():
                        param.trainable = True
                    else:
                        warnings.warn(
                            'The current configs results in a non-parametric model, '
                            'which is no trainable. '
                            'You may need to specify `output_dim` or `embedding_layer_name`.'
                        )

            if (
                _relative_idx_to_embedding_layer
                and _relative_idx_to_embedding_layer >= 1
            ):
                if _relative_idx_to_embedding_layer == 1 and output_dim:
                    replaced_layer = nn.Linear(
                        in_features=_embed_layer['output_features'],
                        out_features=output_dim,
                    )
                    _is_dense_layer_added = True
                else:
                    replaced_layer = _Identity()

                if (
                    '.' in name
                ):  # Note: in torchvision, nested layer names are named with '.' e.g. classifier.0
                    nested_module, layer = name.split('.')
                    setattr(getattr(model, nested_module), layer, replaced_layer)
                else:
                    setattr(model, name, replaced_layer)

            if _relative_idx_to_embedding_layer is not None:
                _relative_idx_to_embedding_layer += 1

        if output_dim and not _is_dense_layer_added:
            # the dense layer needs to be added after the last layer
            model = _LinearAtLast(
                model,
                in_features=_embed_layer['output_features'],
                out_features=output_dim,
            )

        return model


class _Identity(nn.Layer):
    """A placeholder identity operator that is argument-insensitive."""

    def forward(self, input_: Tensor) -> Tensor:
        return input_


class _LinearAtLast(nn.Layer):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self._model = model
        self._linear = nn.Linear(*args, **kwargs)

    def forward(self, input):
        return self._linear(self._model(input))
