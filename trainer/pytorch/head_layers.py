import abc

import torch.nn as nn
import torch.nn.functional as F


class HeadLayer(nn.Module):
    default_loss: nn.Module  #: the recommended loss function to be used when equipping this layer to base model
    arity: int

    def __init__(self, arity_model: nn.Module):
        super().__init__()
        self._arity_model = arity_model

    def forward(self, *inputs):
        return self.call(*self._arity_model(*inputs))

    @abc.abstractmethod
    def call(self, *args, **kwargs):
        ...


class CosineLayer(HeadLayer):
    default_loss = nn.MSELoss()
    arity = 2

    def call(self, lvalue, rvalue):
        return F.cosine_similarity(lvalue, rvalue)


class TripletLayer(HeadLayer):
    default_loss = 'mse'
    arity = 3

    def __init__(self, arity_model: nn.Module, margin: float = 1.0):
        super().__init__(arity_model)
        self._margin = margin

    def call(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, self._margin)
