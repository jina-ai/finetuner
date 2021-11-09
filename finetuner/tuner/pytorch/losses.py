from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseLoss
from ..dataset import ClassDataset, SessionDataset
from .miner import SiameseMiner, SiameseSessionMiner, TripletMiner, TripletSessionMiner


def get_distance(embeddings: torch.Tensor, distance: str) -> torch.Tensor:
    """Get a matrix of pairwise distances between the embedings"""

    if distance == 'cosine':
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        dists = 1 - torch.mm(emb_norm, emb_norm.transpose(0, 1))
    elif distance == 'euclidean':
        dists = torch.cdist(embeddings, embeddings, p=2)
    elif distance == 'sqeuclidean':
        dists = torch.cdist(embeddings, embeddings, p=2) ** 2

    return dists


class SiameseLoss(nn.Module, BaseLoss):
    """Computes the loss for a siamese network.

    The loss for a pair of objects equals ::

        is_sim * dist + (1 - is_sim) * max(0, margin - dist)

    where ``is_sim`` equals 1 if the two objects are similar, and 0 if they are not
    similar. The ``dist`` refers to the distance between the two objects, and ``margin``
    is a number to help bound the loss for dissimilar objects.

    The final loss is the average over losses for all pairs given by the indices.
    """

    def __init__(self, distance: str = 'cosine', margin: float = 1.0):
        """Initialize the loss instance

        :param distance: The type of distance to use, avalilable options are
            ``"cosine"``, ``"euclidean"`` and ``"sqeuclidean"``
        :param margin: The margin to use in loss calculation
        """
        super().__init__()
        self.distance = distance
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the loss

        :param embeddings: An ``[N, d]`` tensor of embeddings
        :param indices: A list of tuple indices and target, where each element in the
            list contains three elements: the indices of the two objects in the pair,
            and their similarity (which equals 1 if they are similar, and 0 if they
            are dissimilar)
        """
        ind_one, ind_two, target = indices
        dist_matrix = get_distance(embeddings, self.distance)
        dists = dist_matrix[ind_one, ind_two]
        target = target.to(torch.float32)

        loss = target * dists + (1 - target) * F.relu(self.margin - dists)
        return loss.mean()

    def get_default_miner(
        self, dataset: Union[ClassDataset, SessionDataset]
    ) -> Union[SiameseMiner, SiameseSessionMiner]:
        if isinstance(dataset, ClassDataset):
            return SiameseMiner()
        elif isinstance(dataset, SessionDataset):
            return SiameseSessionMiner()


class TripletLoss(nn.Module, BaseLoss):
    """Compute the loss for a triplet network.

    The loss for a single triplet equals::

        max(dist_pos - dist_neg + margin, 0)

    where ``dist_pos`` is the distance between the anchor embedding and positive
    embedding, ``dist_neg`` is the distance between the anchor and negative embedding,
    and ``margin`` represents a wedge between the desired anchor-negative and
    anchor-positive distances.

    The final loss is the average over losses for all triplets given by the indices.
    """

    def __init__(self, distance: str = "cosine", margin: float = 1.0):
        """Initialize the loss instance

        :param distance: The type of distance to use, avalilable options are
            ``"cosine"``, ``"euclidean"`` and ``"sqeuclidean"``
        :param margin: The margin to use in loss calculation
        """
        super().__init__()
        self.distance = distance
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the loss

        :param embeddings: An ``[N, d]`` tensor of embeddings
        :param indices: A list of tuple indices, where each element in the list
            contains three elements: the index of anchor, positive match and negative
            match in the embeddings tensor
        """
        ind_anch, ind_pos, ind_neg = indices

        dist_matrix = get_distance(embeddings, self.distance)
        dist_pos = dist_matrix[ind_anch, ind_pos]
        dist_neg = dist_matrix[ind_anch, ind_neg]
        loss = F.relu(dist_pos - dist_neg + self.margin)

        return loss.mean()

    def get_default_miner(
        self, dataset: Union[ClassDataset, SessionDataset]
    ) -> Union[TripletMiner, TripletSessionMiner]:
        if isinstance(dataset, ClassDataset):
            return TripletMiner()
        elif isinstance(dataset, SessionDataset):
            return TripletSessionMiner()
