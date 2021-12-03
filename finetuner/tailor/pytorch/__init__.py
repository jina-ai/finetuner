import copy
from collections import OrderedDict
from typing import Optional, List, TYPE_CHECKING, Union

import numpy as np
import torch
from torch import nn

from ..base import BaseTailor
from ...helper import is_seq_int

if TYPE_CHECKING:
    from ...helper import LayerInfoType, AnyDNN


class PytorchTailor(BaseTailor):
    """Tailor class for PyTorch DNN models"""

    def summary(self, skip_identity_layer: bool = False) -> 'LayerInfoType':
        """Interpret the DNN model and produce model information.

        :param skip_identity_layer: If skip identity layer.
        :return: The model information stored as dict.
        """
        if not self._input_size:
            raise ValueError(
                f'{self.__class__} requires a valid `input_size`, but receiving {self._input_size}'
            )

        user_model = copy.deepcopy(self._model)
        dtypes = [getattr(torch, self._input_dtype)] * len(self._input_size)
        depth = len(list(user_model.modules()))
        for name, module in user_model.named_modules():
            module.name = name

        def _get_shape(output):
            if isinstance(output, (list, tuple)):
                output_shape = [_get_shape(o) for o in output]
                if len(output) == 1:
                    output_shape = output_shape[0]
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
                summary[m_key]['output_shape'] = _get_shape(output)
                summary[m_key]['input_shape'] = _get_shape(input)
                summary[m_key]['module_name'] = module.name

                params = 0
                summary[m_key]['trainable'] = False
                if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                    params += np.prod(list(module.weight.size()))
                    summary[m_key]['trainable'] = module.weight.requires_grad
                if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                    params += np.prod(list(module.bias.size()))
                if hasattr(module, 'all_weights'):
                    params += sum(
                        np.prod(ww.size()) for w in module.all_weights for ww in w
                    )

                summary[m_key]['nb_params'] = params

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and (module != user_model or depth < 1)
            ):
                hooks.append(module.register_forward_hook(hook))

        x = [
            torch.rand(1, *in_size).type(dt)
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
            input_shape = summary[layer]['input_shape']
            is_embedding_layer = not (
                not output_shape
                or not is_seq_int(output_shape)
                or summary[layer]['cls_name'] == self._model.__class__.__name__
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
        freeze: Union[bool, List[str]] = False,
        bottleneck_net: Optional[nn.Module] = None,
    ) -> 'AnyDNN':
        """Convert a general model from :py:attr:`.model` to an embedding model.

        :param layer_name: the name of the layer that is used for output embeddings. All layers *after* that layer
            will be removed. When set to ``None``, then the last layer listed in :py:attr:`.embedding_layers` will be used.
            To see all available names you can check ``name`` field of :py:attr:`.embedding_layers`.
        :param freeze: if set as True, will freeze all layers before :py:`attr`:`layer_name`. If set as list of str, will freeze layers by names.
        :param bottleneck_net: Attach a bottleneck net at the end of model, this module should always trainable.
        :return: Converted embedding model.
        """

        model = copy.deepcopy(self._model)
        _all_embed_layers = {l['name']: l for l in self.embedding_layers}
        if layer_name:
            try:
                _embed_layer = _all_embed_layers[layer_name]
            except KeyError as e:
                raise KeyError(
                    f'`embedding_layer_name` must be one of {_all_embed_layers.keys()}, given {layer_name}'
                ) from e
        else:
            # when not given, using the last layer
            _embed_layer = self.embedding_layers[-1]

        if isinstance(freeze, list):
            # freeze specific layers defined in `freeze_layers`
            for layer_name, param in zip(_all_embed_layers, model.parameters()):
                if layer_name in freeze:
                    param.requires_grad = False
        elif isinstance(freeze, bool) and freeze is True:
            # freeze all layers, not including bottleneck module
            for param in model.parameters():
                param.requires_grad = False

        _embed_layer_output_shape = 0
        _relative_idx_to_embedding_layer = None
        for name, module in model.named_modules():
            if name == _embed_layer['module_name']:
                _relative_idx_to_embedding_layer = 0
                _embed_layer_output_shape = _embed_layer['output_shape']
            if (
                _relative_idx_to_embedding_layer
                and _relative_idx_to_embedding_layer >= 1
            ):
                replaced_layer = nn.Identity(_embed_layer_output_shape)
                if (
                    '.' in name
                ):  # Note: in torchvision, nested layer names are named with '.' e.g. classifier.0
                    nested_module, layer = name.split('.')
                    setattr(getattr(model, nested_module), layer, replaced_layer)
                else:
                    setattr(model, name, replaced_layer)

            if _relative_idx_to_embedding_layer is not None:
                _relative_idx_to_embedding_layer += 1

        if bottleneck_net:
            return nn.Sequential(
                model,
                bottleneck_net,
            )
        return model
