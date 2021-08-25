from typing import Union, Optional
import paddle
from paddle import nn
from paddle.vision.models import mobilenetv2

from .. import PretrainedModelMixin


class MobileNet(nn.Layer, PretrainedModelMixin):
    model_name = 'mobilenetv2'

    def __init__(
        self,
        pretrained_model: Union[str, bool] = False,
        bottleneck_layer: Optional[int] = None,
        freeze_layers: bool = False,
        scale: float = 1.0,
        num_classes: int = 1000,
        with_pool: bool = True,
        **kwargs,
    ):
        super(MobileNet, self).__init__()
        if pretrained_model is None:
            pretrained_model = False

        self._pretrained_model = pretrained_model
        self._base_model = mobilenetv2.mobilenet_v2(
            pretrained=False, scale=scale, **kwargs
        )

        self._num_classes = num_classes
        self._with_pool = with_pool
        self._scale = scale
        if with_pool:
            self.pool2d_avg = nn.AdaptiveAvgPool2D(1)

        self._output_class_logits = True
        if bottleneck_layer is not None:
            self._output_class_logits = False
            assert bottleneck_layer >= -19 & bottleneck_layer < 19

        self._bottleneck_layer = bottleneck_layer
        self._freeze_layers = freeze_layers

        if pretrained_model:
            self.load_pretrained()

        if freeze_layers:
            self.freeze_layers()

        self.base_model.features = nn.Sequential(
            *[self.get_bottleneck_layer(i) for i in range(self.output_layer)]
        )

    def forward(self, x):
        x = self.base_model.features(x)
        if self._with_pool:
            x = self.pool2d_avg(x)

        if self._output_class_logits and self.base_model.num_classes > 0:
            if self.base_model.with_pool:
                x = self.base_model.pool2d_avg(x)
            x = paddle.flatten(x, 1)
            x = self.base_model.classifier(x)
        else:
            x = paddle.flatten(x, 1)

        return x

    @property
    def bottleneck_layers(self):
        return {
            name: layer for name, layer in self.base_model.features.named_children()
        }

    def get_bottleneck_layer(self, index: int):
        assert index <= 18

        return self.bottleneck_layers[str(index)]

    @property
    def output_layer(self):
        if self._bottleneck_layer < 0:
            return 19 + self._bottleneck_layer

        return self._bottleneck_layer

    def load_pretrained(self):
        weight_path = None
        if self._pretrained_model is True:
            model_name = f'{self.model_name}_{self._scale}'
            assert (
                model_name in mobilenetv2.model_urls
            ), f'{model_name} model do not have a pretrained model now'
            model_path, md5sum = mobilenetv2.model_urls[model_name]
            weight_path = paddle.utils.download.get_weights_path_from_url(
                model_path, md5sum
            )
        elif self._pretrained_model:
            weight_path = self._pretrained_model
        else:
            raise ValueError('Please provide available pretrained_model')

        params = paddle.load(weight_path)
        self.base_model.set_dict(params)

    @property
    def input_spec(self):
        from paddle.static import InputSpec

        x_spec = InputSpec(shape=[None, 3, 224, 224], name='x')
        return [x_spec]

    # def flat_model(self) -> nn.Layer:
    #     """Unpack the model architecture recursively and rebuild the model.
    #     :return: Flattened model.
    #     ..note::
    #         Even if we rebuild :attr:`model` into :attr:`flat_model`, weight remains
    #         the same at layer level.
    #     """
    #
    #     print(f'=> bottles: {list(self.bottleneck_layers.keys())}')
    #     print(f'=> bottle at 1: {self.get_bottleneck_layer(1)}')
