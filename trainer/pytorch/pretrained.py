from typing import List

import torch.nn as nn
import torchvision.models as models

from ..pretrained import ModelInterpreter


class TorchModelInterpreter(ModelInterpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = getattr(models, self._model_name)(pretrained=True)

    def _interpret_linear_layers(self):
        """Mapping of linear layer index and a layer represented as :class:`nn.Module`.

        This property get the last n layers and build a map between layer index
        and layer instance. The layer index will be represented as negative numbers,
        e.g. -1 means the last layer is linear layer.
        """
        rv = {}
        if isinstance(self._last_layer, nn.Sequential):  # last layer is nested.
            modules = list(self._last_layer.modules())
            for index, module in enumerate(modules):
                if isinstance(module, nn.Linear):
                    rv[
                        index - len(modules)
                    ] = module  # reverse the order, e.g. -1 last layer
        else:
            modules = list(self.base_model.modules())
            for index, module in enumerate(self.base_model.modules()):
                if isinstance(module, nn.Linear):
                    rv[index - len(modules)] = module
        return rv

    @property
    def _last_layer(self):
        return list((self.base_model.children()))[-1]

    @_last_layer.setter
    def _last_layer(self, other_layer: nn.Module):
        self._last_layer = other_layer

    def _chop_off_last_n_layers(self, layer_index: int):
        """Modify base_model based on :attr:`layer_name`.

        Remove last n layers given the layer index, and replace current layer with :class:`nn.Linear`
            with the new :attr:`out_features` as dimensionality.
        :param layer_index: the layer index (a negative number )to remove.
        :return: Modified model.
        """
        index_layer_map = self._interpret_linear_layers()
        chopped_layer = index_layer_map[layer_index]
        out_layer = nn.Linear(
            in_features=chopped_layer.out_features,
            out_features=self._out_features or chopped_layer.out_features,
            bias=self._bias,
        )
        model = self.base_model
        if isinstance(self._last_layer, nn.Sequential):
            modules = list(self._last_layer[:layer_index])
            modules.append(out_layer)
            model.classifier = nn.Sequential(*modules)
        else:
            model.fc = out_layer
        return model

    @property
    def trainable_layers(self) -> List[int]:
        """Get trainable layers, e.g. names Linear layers in the backbone model.

        :return: List of linear layer names.
        """
        return list(self._interpret_linear_layers().keys())

    def get_modified_base_model(self, layer_index: int) -> nn.Module:
        """Modify base model based on :attr:`layer_name`. E.g. remove the last n layers
            for retrain.
        :param layer_index: Layer name to modify, if specified, all later layers
            will be removed, the :attr:`out_features` of current layer will be replaced
            with the new output dimension.
        :return: The new base model to be trained.
        """
        if layer_index not in self.trainable_layers:
            msg = f'Layer index {layer_index} is not a valid layer in your model.'
            msg += f'expect one of the layer in {self.trainable_layers}.'
            raise ValueError(msg)
        return self._chop_off_last_n_layers(layer_index)
