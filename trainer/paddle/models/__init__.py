import abc
import paddle
from paddle import nn


def freeze_params(model: nn.Layer):
    for param in model.parameters():
        param.trainable = False


class PretrainedModelMixin:
    @abc.abstractmethod
    def load_pretrained(self, model_path: str):
        ...

    @property
    @abc.abstractmethod
    def input_spec(self):
        ...

    @property
    def base_model(self):
        return self._base_model

    def freeze_layers(self):
        if self.base_model:
            freeze_params(self.base_model)

    def to_static(self):
        return paddle.jit.to_static(self.base_model, input_spec=self.input_spec)
