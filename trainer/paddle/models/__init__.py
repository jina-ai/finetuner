import abc
from typing import Union
from paddle import nn


def freeze_params(model: nn.Layer):
    for param in model.parameters():
        param.trainable = False


class PretrainedModelMixin:
    @abc.abstractmethod
    def load_pretrained(self, model_path: str):
        ...

    @abc.abstractmethod
    def to_static(self):
        ...
