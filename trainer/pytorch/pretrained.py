from typing import Optional, List

import torch.nn as nn
import torchvision.models as models


class ModelInterpreter:
    def __init__(
        self, model_name: str, out_features: Optional[int], freeze=True, bias=True
    ):
        self.base_model = getattr(models, model_name)(pretrained=freeze)
        self._interpret_linear_layers()
        self.out_features = out_features
        self.bias = bias

    def _interpret_linear_layers(self):
        """Get all Linear layers inside a model.

        Pytorch use named_modules to get layers recursively.
        :return rv: Dict indicates the layer names and Layer itself,
            to keep the dimensionality of the new layer.
        """
        rv = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                rv[name] = module
        return rv

    def _chop_off_last_n_layers(self, layer_name: str):
        """Modify base_model in place based on :attr:`layer_name`.

        For a pytorch application it normally consist of 2 cases.
        First, a Linear layer is wrapped by one or multiple :class:`nn.Sequential`.
           In this case, all Linear layers has been unpacked inside :meth:`_interpret_linear_layers`.
           And the name of layer consist of 2 parts: [MODULE-NAME.LAYER_INDEX]
        """
        name_layer_map = self._interpret_linear_layers()
        if '.' in layer_name:
            module_name, layer_idx = layer_name.split('.')
            module = getattr(self.base_model, module_name)
            module = module[: int(layer_idx)]  # remove all layers after layer_idx
            module.add_module(
                module_name,
                nn.Linear(
                    in_features=name_layer_map[layer_name].in_features,
                    out_features=self.out_features
                    or name_layer_map[layer_name].out_features,
                    bias=self.bias,
                ),
            )
            setattr(self.base_model, module_name, module)
        else:
            # if . not included in layer_name, means it's not wrapped, and it's the last layer.
            # no need to remove
            setattr(
                self.base_model,
                layer_name,
                nn.Linear(
                    in_features=name_layer_map[layer_name].in_features,
                    out_features=self.out_features
                    or name_layer_map[layer_name].out_features,
                    bias=self.bias,
                ),
            )

    @property
    def trainable_layers(self) -> List[str]:
        """Get trainable layers, e.g. names Linear layers in the backbone model.

        :return: List of linear layer names.
        """
        return list(self._interpret_linear_layers().keys())

    def modify_base_model(self, layer_name: str) -> nn.Module:
        """Modify base model based on :attr:`layer_name`. E.g. remove the last n layers
            for retrain.
        :param layer_name: Layer name to modify, if specified, all later layers
            will be removed, the :attr:`out_features` of current layer will be replaced
            with the new output dimension.
        :return: The new base model to be trained.
        """
        if layer_name not in self.trainable_layers:
            msg = f'Layer name {layer_name} is not a valid layer in your model.'
            msg += f'expect one of the layer in {self.trainable_layers}.'
            raise ValueError(msg)
        self._chop_off_last_n_layers(layer_name)
        return self.base_model
