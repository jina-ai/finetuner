from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_dist(emb_one: torch.Tensor, emb_two: torch.Tensor) -> torch.Tensor:
    return 1 - F.cosine_similarity(emb_one, emb_two)


def _euclidean_dist(emb_one: torch.Tensor, emb_two: torch.Tensor) -> torch.Tensor:
    return F.pairwise_distance(emb_one, emb_two, p=2)


def _dist_fn(dist_name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if dist_name == 'cosine':
        return _cosine_dist
    elif dist_name == 'euclidean':
        return _euclidean_dist


class SiameseLoss(nn.Module):
    """Computes the loss for a siamese network.

    The loss for a pair of objects equals ::

        0.5 * ( is_sim * dist + (1 - is_sim) * max(0, margin - dist) )^2

    where ``is_sim`` equals 1 if the two objects are similar, and 0 if they are not
    similar. The ``dist`` refers to the distance between the two objects, and ``margin``
    is a number to help bound the loss for dissimilar objects.

    The final loss is the average over losses for all pairs given by the indices.
    """

    def __init__(self, distance: str = "cosine", margin: float = 0.0):
        """Initialize the loss instance

        :param distance: The type of distance to use, avalilable options are
            ``"cosine"`` and ``"euclidean"``
        :param margin: The margin to use in loss calculation
        """

        self.distance = distance
        self.margin = margin

    def forward(
        self, embeddings: torch.Tensor, indices: List[Tuple[int, int, int]]
    ) -> torch.Tensor:
        """Compute the loss

        :param embeddings: An ``[N, d]`` tensor of embeddings
        :param indices: A list of tuple indices and target, where each element in the
            list contains three elements: the indices of the two objects in the pair,
            and their similarity (which equals 1 if they are similar, and 0 if they
            are dissimilar)
        """
        ind_one, ind_two, target = list(zip(*indices))

        target = torch.tensor(target, dtype=torch.float32, device=embeddings.device)
        emb_one, emb_two = embeddings[ind_one], embeddings[ind_two]
        dist = _dist_fn(self.distance)(emb_one, emb_two)

        loss = 0.5 * (target * dist + (1 - target) * F.relu(0, self.margin - dist)) ** 2
        return loss.mean()


class TripletLoss(nn.Module):
    """Compute the loss for a triplet network.

    The loss for a single triplet equals::

        max(dist_pos - dist_neg + margin, 0)

    where ``dist_pos`` is the distance between the anchor embedding and positive
    embedding, ``dist_neg`` is the distance between the anchor and negative embedding,
    and ``margin`` represents a wedge between the desired anchor-negative and
    anchor-positive distances.

    The final loss is the average over losses for all triplets given by the indices.
    """

    def __init__(self, distance: str = "cosine", margin: float = 0.0):
        """Initialize the loss instance

        :param distance: The type of distance to use, avalilable options are
            ``"cosine"`` and ``"euclidean"``
        :param margin: The margin to use in loss calculation
        """
        self.distance = distance
        self.margin = margin

    def forward(
        self, embeddings: torch.Tensor, indices: List[Tuple[int, int, int]]
    ) -> torch.Tensor:
        """Compute the loss

        :param embeddings: An ``[N, d]`` tensor of embeddings
        :param indices: A list of tuple indices, where each element in the list
            contains three elements: the index of anchor, positive match and negative
            match in the embeddings tensor
        """
        ind_anch, ind_pos, ind_neg = list(zip(*indices))
        emb_anch, emb_pos, emb_neg = (
            embeddings[ind_anch],
            embeddings[ind_pos],
            embeddings[ind_neg],
        )
        dist_pos = _dist_fn(self.distance)(emb_anch, emb_pos)
        dist_neg = _dist_fn(self.distance)(emb_anch, emb_neg)

        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()
