import torch.nn as nn
import torchvision.models as models


class ModelInterpreter:
    def __init__(
        self, model_name: str, layer_name: str, out_features: int = 32, pretrained=True
    ):
        self.base_model = getattr(models, model_name)(pretrained=pretrained)
        self._interpret_linear_layers()
        if layer_name not in self.trainable_layers:
            msg = f'Layer name {layer_name} is not a valid layer in your model.'
            msg += f'expect one of the layer in {self.trainable_layers}.'
            raise ValueError(msg)
        self.layer_name = layer_name
        self.out_features = out_features
        self._chop_off_last_n_layers()

    def _interpret_linear_layers(self):
        rv = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                rv[name] = module.in_features
        return rv

    def _chop_off_last_n_layers(self):
        """Modify base_model in place."""
        name_feature_map = self._interpret_linear_layers()
        if '.' in self.layer_name:
            module_name, layer_idx = self.layer_name.split('.')
            module = getattr(self.base_model, module_name)
            current_layer = module[int(layer_idx)]
            module = module[: int(layer_idx)]
            module.add_module(
                'linear_final',
                nn.Linear(
                    in_features=current_layer.in_features,
                    out_features=self.out_features,
                ),
            )
            setattr(self.base_model, module_name, module)
        else:
            # if . not included in layer_name, means it's not wrapped, and it's the last layer.
            # no need to remove
            setattr(
                self.base_model,
                self.layer_name,
                nn.Linear(
                    in_features=name_feature_map[self.layer_name],
                    out_features=self.out_features,
                ),
            )

    @property
    def trainable_layers(self):
        return self._interpret_linear_layers().keys()

    def interpret(self):
        self._chop_off_last_n_layers()
        return self.base_model


# model = ModelInterpreter('resnet18', layer_name='fc')
# print(model._chop_off_last_n_layers())
#
#
# model = ModelInterpreter('alexnet', layer_name='classifier.1')
# print(model._chop_off_last_n_layers())
