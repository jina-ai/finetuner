import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseHead


class CosineSiameseHead(BaseHead, nn.Module):
    arity = 2

    def get_output(self, lvalue, rvalue):
        return F.cosine_similarity(lvalue, rvalue)

    def metric_fn(self, pred_val, target_val):
        s = torch.count_nonzero(torch.eq(torch.sign(pred_val), torch.sign(target_val)))
        return s / len(pred_val)

    def loss_fn(self, pred_val, target_val):
        return F.mse_loss(pred_val, target_val)


class EuclideanSiameseHead(BaseHead, nn.Module):
    arity = 2

    def __init__(self, arity_model: nn.Module, margin: float = 0.7):
        super().__init__(arity_model=arity_model)
        self.margin = margin

    def get_output(self, lvalue, rvalue):
        return torch.square(lvalue - rvalue).sum(dim=-1)

    def metric_fn(self, pred_val, target_val):
        s = torch.count_nonzero(torch.eq(torch.sign(pred_val), torch.sign(target_val)))
        return s / len(pred_val)

    def loss_fn(self, pred_val, target_val):
        is_similar = target_val > 0
        loss = torch.square(
            is_similar * pred_val + (1 - is_similar) * F.relu(self.margin - pred_val)
        )

        return loss.mean()


class EuclideanTripletHead(BaseHead, nn.Module):
    arity = 3

    def __init__(self, arity_model: nn.Module, margin: float = 1.0):
        super().__init__(arity_model)
        self._margin = margin

    def get_output(self, anchor, positive, negative):
        dist_pos = torch.square(anchor - positive).sum(dim=-1)
        dist_neg = torch.square(anchor - negative).sum(dim=-1)
        return dist_pos, dist_neg

    def metric_fn(self, pred_val, target_val):
        dist_pos, dist_neg = pred_val

        s = torch.sum(torch.less(dist_pos, dist_neg))
        return s / len(target_val)

    def loss_fn(self, pred_val, target_val):
        dist_pos, dist_neg = pred_val
        return torch.mean(F.relu(dist_pos - dist_neg + self._margin))


class EuclideanCosineHead(EuclideanTripletHead):
    arity = 3

    def get_output(self, anchor, positive, negative):
        dist_pos = 1 - F.cosine_similarity(anchor, positive)
        dist_neg = 1 - F.cosine_similarity(anchor, negative)
        return dist_pos, dist_neg
