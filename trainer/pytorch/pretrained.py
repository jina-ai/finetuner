from typing import List

import torch.nn as nn
import torchvision.models as models

from ..pretrained import ModelInterpreter


class TorchModelInterpreter(ModelInterpreter):
    def __init__(self):
        super().__init__()
        self.base_model = getattr(models, self._model_name)(pretrained=self._freeze)
        self._flat_model = None

    @property
    def flat_model(self) -> nn.Module:
        """Unpack the model architecture recursively and rebuild the model.

        :return: Flattened model.
        """
        if not self._flat_model:
            modules = []
            for module in self.base_model.modules():
                if not isinstance(module, (nn.Sequential, type(self.base_model))):
                    modules.append(module)
            self._flat_model = nn.Sequential(*modules)
        return self._flat_model

    @flat_model.setter
    def flat_model(self, other_model: nn.Module):
        """Unpack the model architecture recursively and rebuild the model.

        :param other_model: Set the current flattened model as other model.
        """
        self._flat_model = other_model

    def _interpret_linear_layers(self):
        """Get all Linear layers inside a model.

        Pytorch use named_modules to get layers recursively.
        :return rv: Dict indicates the layer names and Layer itself,
            to keep the dimensionality of the new layer.
        """
        rv = {}
        for name, module in self.flat_model.named_modules():
            if isinstance(module, nn.Linear):
                rv[int(name)] = module
        return rv

    def _chop_off_last_n_layers(self, layer_index: int):
        """Modify base_model in place based on :attr:`layer_name`.

        Remove last n layers given the layer index, and replace current layer with :class:`nn.Linear`
            with the new :attr:`out_features` as dimensionality.
        :param layer_index: the layer index to remove.
        :return: Modified model.
        """
        name_layer_map = self._interpret_linear_layers()
        self.flat_model = self.flat_model[:layer_index]
        setattr(
            self.flat_model,
            str(layer_index),
            nn.Linear(
                in_features=name_layer_map[layer_index].in_features,
                out_features=self._out_features
                or name_layer_map[layer_index].out_features,
                bias=self._bias,
            ),
        )
        return self.flat_model

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
