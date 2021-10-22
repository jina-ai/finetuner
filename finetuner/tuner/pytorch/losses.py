from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseLoss


class CosineSiameseLoss(BaseLoss, nn.Module):
    """Computes the loss for a siamese network using cosine distance.

    The loss for a pair of objects equals ``(target - cos_sim)^2``, where ``target``
    should equal 1 when both objects belong to the same class, and to -1 when they
    belong to different classes. The ``cos_sim`` represents the cosime similarity
    between both objects.

    The final loss is the average over losses for all pairs of objects in the batch.
    """

    arity = 2

    def forward(
        self, embeddings: List[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss.

        :param embeddings: Should be a list or a tuple containing two tensors:
            - ``[N, D]`` tensor of embeddings of the first objects of the pair
            - ``[N, D]`` tensor of embeddings of the second object of the pair
        :param target: A ``[N, ]`` tensor of target values
        """
        l_emb, r_emb = embeddings
        cos_sim = F.cosine_similarity(l_emb, r_emb)
        loss = F.mse_loss(cos_sim, target)
        return loss


class EuclideanSiameseLoss(BaseLoss, nn.Module):
    """Computes the loss for a siamese network using eculidean distance.

    This loss is also known as contrastive loss.

    The loss being optimized equals::

        [is_sim * dist + (1 - is_sim) * max(margin - dist, 0)]^2

    where ``target`` should equal 1 when both objects belong to the same class,
    and 0 otheriwse. The ``dist`` is the euclidean distance between the embeddings of
    the objects, and ``margin`` is some number, used here to ensure better stability
    of training.

    The final loss is the average over losses for all pairs of objects in the batch.
    """

    arity = 2

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, embeddings: List[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss.

        :param inputs: Should be a list or a tuple containing three tensors:
            - ``[N, D]`` tensor of embeddings of the first objects of the pair
            - ``[N, D]`` tensor of embeddings of the second objects of the pair
        :param target: A ``[N, ]`` tensor of target values
        """
        l_emb, r_emb = embeddings
        eucl_dist = F.pairwise_distance(l_emb, r_emb, p=2)
        is_similar = (target > 0).to(torch.float32)

        loss = 0.5 * torch.square(
            is_similar * eucl_dist + (1 - is_similar) * F.relu(self.margin - eucl_dist)
        )
        return loss.mean()


class EuclideanTripletLoss(BaseLoss, nn.Module):
    """Compute the loss for a triplet network using euclidean distance.

    The loss is computed as ``max(dist_pos - dist_neg + margin, 0)``, where ``dist_pos``
    is the euclidean distance between the anchor embedding and positive embedding,
    ``dist_neg`` is the euclidean distance between the anchor and negative embedding,
    and ``margin`` represents a wedge between the desired wedge between anchor-negative
    and anchor-positive distances.

    The final loss is the average over losses for all triplets in the batch.
    """

    arity = 3

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self._margin = margin

    def forward(
        self, embeddings: List[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss.

        :param inputs: Should be a list or a tuple containing three tensors:
            - ``[N, D]`` tensor of embeddings of the anchor objects
            - ``[N, D]`` tensor of embeddings of the positive objects
            - ``[N, D]`` tensor of embeddings of the negative objects
        """
        anchor, positive, negative = embeddings
        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        dist_neg = F.pairwise_distance(anchor, negative, p=2)

        return torch.mean(F.relu(dist_pos - dist_neg + self._margin))


class CosineTripletLoss(EuclideanTripletLoss):
    """Compute the loss for a triplet network using cosine distance.

    The loss is computed as ``max(dist_pos - dist_neg + margin, 0)``, where ``dist_pos``
    is the cosine distance between the anchor embedding and positive embedding,
    ``dist_neg`` is the cosine distance between the anchor and negative embedding, and
    ``margin`` represents a wedge between the desired wedge between anchor-negative and
    anchor-positive distances.

    The final loss is the average over losses for all triplets in the batch.
    """

    def forward(
        self, embeddings: List[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss.

        :param inputs: Should be a list or a tuple containing three tensors:
            - ``[N, D]`` tensor of embeddings of the anchor objects
            - ``[N, D]`` tensor of embeddings of the positive objects
            - ``[N, D]`` tensor of embeddings of the negative objects
        """
        anchor, positive, negative = embeddings
        dist_pos = 1 - F.cosine_similarity(anchor, positive)
        dist_neg = 1 - F.cosine_similarity(anchor, negative)

        return torch.mean(F.relu(dist_pos - dist_neg + self._margin))
