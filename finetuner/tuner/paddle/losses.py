from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..base import BaseLoss


class CosineSiameseLoss(BaseLoss, nn.Layer):
    arity = 2

    def forward(
        self, embeddings: List[paddle.Tensor], target: paddle.Tensor
    ) -> paddle.Tensor:
        l_emb, r_emb = embeddings
        cos_sim = F.cosine_similarity(l_emb, r_emb)
        loss = F.mse_loss(cos_sim, target)
        return loss


class EuclideanSiameseLoss(BaseLoss, nn.Layer):
    arity = 2

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self._dist = nn.PairwiseDistance(2)

    def forward(
        self, embeddings: List[paddle.Tensor], target: paddle.Tensor
    ) -> paddle.Tensor:
        l_emb, r_emb = embeddings
        eucl_dist = self._dist(l_emb, r_emb)
        is_similar = paddle.cast(target > 0, paddle.float32)

        loss = 0.5 * paddle.square(
            is_similar * eucl_dist + (1 - is_similar) * F.relu(self.margin - eucl_dist)
        )
        return loss.mean()


class EuclideanTripletLoss(BaseLoss, nn.Layer):
    arity = 3

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self._margin = margin
        self._dist = nn.PairwiseDistance(2)

    def forward(
        self, embeddings: List[paddle.Tensor], target: paddle.Tensor
    ) -> paddle.Tensor:
        anchor, positive, negative = embeddings
        dist_pos = self._dist(anchor, positive)
        dist_neg = self._dist(anchor, negative)

        return paddle.mean(F.relu(dist_pos - dist_neg + self._margin))


class CosineTripletLoss(BaseLoss, nn.Layer):
    arity = 3

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self._margin = margin

    def forward(
        self, embeddings: List[paddle.Tensor], target: paddle.Tensor
    ) -> paddle.Tensor:
        anchor, positive, negative = embeddings
        dist_pos = 1 - F.cosine_similarity(anchor, positive)
        dist_neg = 1 - F.cosine_similarity(anchor, negative)

        return paddle.mean(F.relu(dist_pos - dist_neg + self._margin))
