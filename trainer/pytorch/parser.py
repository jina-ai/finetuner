from typing import List

import torch.nn as nn
import torchvision.models as models

from ..parser import ModelParser


class TorchModelParser(ModelParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = getattr(models, self._model_name)(pretrained=True)

    def _parse_base_model(self):
        def _traverse_flat(model):
            flattened = []
            childs = list(model.children())
            if not childs:
                return model
            else:
                for child in childs:
                    try:
                        flattened.extend(_traverse_flat(child))
                    except TypeError:
                        flattened.append(_traverse_flat(child))
            return flattened

        return _traverse_flat(self.base_model)

    def _chop_off_last_n_layers(self, layer_index: int):
        """Modify base_model based on :attr:`layer_name`.

        Remove last n layers given the layer index, and replace current layer with :class:`nn.Linear`
            with the new :attr:`out_features` as dimensionality.
        :param layer_index: the layer name to remove.
        :return: Modified model.
        """
        modules = self._parse_base_model()
        chopped_layer = modules[layer_index]
        model_extracted = nn.Sequential(*modules[:layer_index])
        if self._freeze:
            for layer in model_extracted.modules():
                layer.require_grad = False
        tail_layer = nn.Linear(
            in_features=chopped_layer.in_features,
            out_features=self._out_features or chopped_layer.out_features,
            bias=self._bias,
        )  # trainable tail layer
        return nn.Sequential(model_extracted, tail_layer)

    def get_modified_base_model(self, layer_index: int) -> nn.Module:
        """Modify base model based on :attr:`layer_name`. E.g. remove the last n layers
            for retrain.
        :param layer_index: Layer name to modify, if specified, all later layers
            will be removed, the :attr:`out_features` of current layer will be replaced
            with the new output dimension.
        :return: The new base model to be trained.
        """
        return self._chop_off_last_n_layers(layer_index)
