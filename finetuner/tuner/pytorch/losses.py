from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseLoss


class CosineSiameseLoss(BaseLoss, nn.Module):
    arity = 2

    def forward(
        self, embeddings: List[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        l_emb, r_emb = embeddings
        cos_sim = F.cosine_similarity(l_emb, r_emb)
        loss = F.mse_loss(cos_sim, target)
        return loss


class EuclideanSiameseLoss(BaseLoss, nn.Module):
    arity = 2

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, embeddings: List[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        l_emb, r_emb = embeddings
        eucl_dist = F.pairwise_distance(l_emb, r_emb, p=2)
        is_similar = (target > 0).to(torch.float32)

        loss = 0.5 * torch.square(
            is_similar * eucl_dist + (1 - is_similar) * F.relu(self.margin - eucl_dist)
        )
        return loss.mean()


class EuclideanTripletLoss(BaseLoss, nn.Module):
    arity = 3

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self._margin = margin

    def forward(
        self, embeddings: List[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        anchor, positive, negative = embeddings
        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        dist_neg = F.pairwise_distance(anchor, negative, p=2)

        return torch.mean(F.relu(dist_pos - dist_neg + self._margin))


class CosineTripletLoss(EuclideanTripletLoss):
    def forward(
        self, embeddings: List[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        anchor, positive, negative = embeddings
        dist_pos = 1 - F.cosine_similarity(anchor, positive)
        dist_neg = 1 - F.cosine_similarity(anchor, negative)

        return torch.mean(F.relu(dist_pos - dist_neg + self._margin))
