import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadLayer(nn.Module):
    arity: int

    def __init__(self, arity_model: nn.Module):
        super().__init__()
        self._arity_model = arity_model

    def forward(self, *inputs):
        args = self._arity_model(*inputs)
        return self.get_output_for_loss(*args), self.get_output_for_metric(*args)

    @abc.abstractmethod
    def get_output_for_loss(self, *inputs):
        ...

    @abc.abstractmethod
    def get_output_for_metric(self, *inputs):
        ...

    @abc.abstractmethod
    def loss_fn(self, pred_val, target_val):
        ...

    @abc.abstractmethod
    def metric_fn(self, pred_val, target_val):
        ...


class CosineLayer(HeadLayer):
    arity = 2

    def __init__(self, arity_model: nn.Module):
        super().__init__(arity_model)
        self._stats = 0
        self._total = 0

    def get_output_for_loss(self, lvalue, rvalue):
        return F.cosine_similarity(lvalue, rvalue)

    def get_output_for_metric(self, lvalue, rvalue):
        return F.cosine_similarity(lvalue, rvalue)

    def metric_fn(self, pred_val, target_val):
        s = torch.count_nonzero(torch.eq(torch.sign(pred_val), torch.sign(target_val)))
        self._stats += s
        self._total += len(pred_val)
        return (self._stats / self._total).numpy()

    def loss_fn(self, pred_val, target_val):
        return F.mse_loss(pred_val, target_val)


class TripletLayer(HeadLayer):
    arity = 3

    def __init__(self, arity_model: nn.Module, margin: float = 1.0):
        super().__init__(arity_model)
        self._margin = margin
        self._stats = 0
        self._total = 0

    def get_output_for_loss(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, self._margin)

    def get_output_for_metric(self, anchor, positive, negative):
        return F.cosine_similarity(anchor, positive), F.cosine_similarity(
            anchor, negative
        )

    def metric_fn(self, pred_val, target_val):
        y_positive, y_negative = pred_val
        s_p = torch.count_nonzero(torch.greater(y_positive, 0))
        s_n = torch.count_nonzero(torch.less(y_negative, 0))
        self._stats += s_p + s_n
        self._total += len(y_positive) + len(y_negative)
        return (self._stats / self._total).numpy()

    def loss_fn(self, pred_val, target_val):
        return F.mse_loss(pred_val, target_val)
