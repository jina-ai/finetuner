from typing import Optional, Union

import paddle
from paddle import nn
from paddle.vision.models import resnet

from .. import PretrainedModelMixin, freeze_params


class ResNet(nn.Layer, PretrainedModelMixin):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    """

    model_name: str = 'resnet'

    def __init__(
        self,
        pretrained_model: Union[str, bool] = False,
        bottleneck_layer: Optional[int] = None,
        freeze_layers: bool = False,
        **kwargs,
    ):
        super(ResNet, self).__init__()
        if pretrained_model is None:
            pretrained_model = False

        self._pretrained_model = pretrained_model
        self._base_model = None

        self._output_class_logits = True
        if bottleneck_layer is not None:
            self._output_class_logits = False
            assert bottleneck_layer >= -4 & bottleneck_layer < 4

        self._bottleneck_layer = bottleneck_layer
        self._freeze_layers = freeze_layers

    @property
    def base_model(self):
        return self._base_model

    @property
    def bottleneck_layers(self):
        return [self.get_bottleneck_layer(i) for i in range(4)]

    @property
    def output_layer(self):
        if self._bottleneck_layer < 0:
            return 4 + self._bottleneck_layer

        return self._bottleneck_layer

    def get_bottleneck_layer(self, index: int):
        assert index < 4
        # Note: `__getattr__` is overided in `nn.Layer`
        return {name: layer for name, layer in self.base_model.named_children()}[
            f'layer{index+1}'
        ]

    def freeze_layers(self):
        if self.base_model:
            freeze_params(self.base_model)

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        for i, layer in enumerate(self.bottleneck_layers):
            x = layer(x)
            # apply early stop
            if (not self._output_class_logits) and self.output_layer == i:
                break

        if self._output_class_logits and (self.base_model.num_classes > 0):
            if self.base_model.with_pool:
                x = self.base_model.avgpool(x)
            x = paddle.flatten(x, 1)
            x = self.base_model.fc(x)
        else:
            x = paddle.flatten(x, 1)

        return x

    def load_pretrained(self):
        weight_path = None
        if self._pretrained_model is True:
            assert (
                self.model_name in resnet.model_urls
            ), f'{self.model_name} model do not have a pretrained model now'
            model_path, md5sum = resnet.model_urls[self.model_name]
            weight_path = paddle.utils.download.get_weights_path_from_url(
                model_path, md5sum
            )
        elif self._pretrained_model:
            weight_path = self._pretrained_model
        else:
            raise ValueError('Please provide available pretrained_model')

        params = paddle.load(weight_path)
        self.base_model.set_dict(params)

    def to_static(self):
        return None


class ResNet18(ResNet):
    model_name = 'resnet18'

    def __init__(
        self,
        pretrained_model: Union[str, bool] = False,
        bottleneck_layer: Optional[int] = None,
        freeze_layers: bool = False,
        **kwargs,
    ):
        super().__init__(
            pretrained_model=pretrained_model,
            bottleneck_layer=bottleneck_layer,
            freeze_layers=freeze_layers,
            **kwargs,
        )
        self._base_model = resnet.resnet18(pretrained=False, **kwargs)
        if pretrained_model:
            self.load_pretrained()
        if freeze_layers:
            self.freeze_layers()

    @property
    def flat_model(self) -> nn.Layer:
        """Unpack the model architecture recursively and rebuild the model.
        :return: Flattened model.
        ..note::
            Even if we rebuild :attr:`model` into :attr:`flat_model`, weight remains
            the same at layer level.
        """
        if not self._flat_model:
            modules = []

            for prefix, layer in self.base_model.named_children():
                print(f'{prefix} -> {layer}')
            # for module in self._base_model.children():
            #
            #     if not isinstance(module, (nn.Sequential, type(self.base_model))):
            #         modules.append(module)
            # self._flat_model = nn.Sequential(*modules)
        # return self._flat_model
        return None

    def to_static(self):
        from paddle.static import InputSpec

        x_spec = InputSpec(shape=[None, 3, 224, 224], name='x')
        return paddle.jit.to_static(self.base_model, input_spec=[x_spec])
