import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseHead


class CosineLayer(BaseHead, nn.Module):
    arity = 2

    def get_output_for_loss(self, lvalue, rvalue):
        return F.cosine_similarity(lvalue, rvalue)

    def get_output_for_metric(self, lvalue, rvalue):
        return F.cosine_similarity(lvalue, rvalue)

    def metric_fn(self, pred_val, target_val):
        s = torch.count_nonzero(torch.eq(torch.sign(pred_val), torch.sign(target_val)))
        return (s / len(pred_val)).numpy()

    def loss_fn(self, pred_val, target_val):
        return F.mse_loss(pred_val, target_val)


class TripletLayer(BaseHead, nn.Module):
    arity = 3

    def __init__(self, arity_model: nn.Module, margin: float = 1.0):
        super().__init__(arity_model)
        self._margin = margin

    def get_output_for_loss(self, anchor, positive, negative):
        return F.triplet_margin_loss(
            anchor, positive, negative, self._margin, reduction='none'
        )

    def get_output_for_metric(self, anchor, positive, negative):
        return F.cosine_similarity(anchor, positive), F.cosine_similarity(
            anchor, negative
        )

    def metric_fn(self, pred_val, target_val):
        y_positive, y_negative = pred_val
        s_p = torch.count_nonzero(torch.greater(y_positive, 0))
        s_n = torch.count_nonzero(torch.less(y_negative, 0))
        return ((s_p + s_n) / (len(y_positive) + len(y_negative))).numpy()

    def loss_fn(self, pred_val, target_val):
        return F.mse_loss(pred_val, target_val)
