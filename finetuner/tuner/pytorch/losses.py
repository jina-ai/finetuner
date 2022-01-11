from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseLoss, BaseMiner
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


class PytorchTupleLoss(nn.Module, BaseLoss[torch.Tensor]):
    """Base class for all pytorch losses."""

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if self.miner is None:
            # If labels is a tuple of tensors, this is a session dataset
            self.miner = self.get_default_miner(isinstance(labels, (list, tuple)))

        dists = get_distance(embeddings, self.distance)
        mined_tuples = self.miner.mine(labels, dists.clone().detach())
        loss = self.compute(embeddings, mined_tuples)

        return loss


class SiameseLoss(PytorchTupleLoss):
    """Computes the loss for a siamese network.

    The loss for a pair of objects equals ::

        is_sim * dist + (1 - is_sim) * max(0, margin - dist)

    where ``is_sim`` equals 1 if the two objects are similar, and 0 if they are not
    similar. The ``dist`` refers to the distance between the two objects, and ``margin``
    is a number to help bound the loss for dissimilar objects.

    The final loss is the average over losses for all pairs given by the indices.
    """

    def __init__(
        self,
        distance: str = 'cosine',
        margin: float = 1.0,
        miner: Optional[BaseMiner] = None,
    ):
        """Initialize the loss instance

        :param distance: The type of distance to use, avalilable options are
            ``"cosine"``, ``"euclidean"`` and ``"sqeuclidean"``
        :param margin: The margin to use in loss calculation
        :param miner: The miner to use. If not provided, a default minuer that
            selects all possible pairs will be used
        """
        super().__init__()
        self.distance = distance
        self.margin = margin
        self.miner = miner

    def compute(
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
        if ind_one.nelement() == 0 or ind_two.nelement() == 0 or target.nelement() == 0:
            raise ValueError('Got empty tuple/triplets from your dataset.')
        dist_matrix = get_distance(embeddings, self.distance)
        dists = dist_matrix[ind_one, ind_two]
        target = target.to(torch.float32)

        loss = target * dists + (1 - target) * F.relu(self.margin - dists)
        return loss.mean()

    def get_default_miner(
        self, is_session_dataset: bool
    ) -> Union[SiameseMiner, SiameseSessionMiner]:
        if not is_session_dataset:
            return SiameseMiner()
        else:
            return SiameseSessionMiner()


class TripletLoss(PytorchTupleLoss):
    """Compute the loss for a triplet network.

    The loss for a single triplet equals::

        max(dist_pos - dist_neg + margin, 0)

    where ``dist_pos`` is the distance between the anchor embedding and positive
    embedding, ``dist_neg`` is the distance between the anchor and negative embedding,
    and ``margin`` represents a wedge between the desired anchor-negative and
    anchor-positive distances.

    The final loss is the average over losses for all triplets given by the indices.
    """

    def __init__(
        self,
        distance: str = 'cosine',
        margin: float = 1.0,
        miner: Optional[BaseMiner] = None,
    ):
        """Initialize the loss instance

        :param distance: The type of distance to use, avalilable options are
            ``"cosine"``, ``"euclidean"`` and ``"sqeuclidean"``
        :param margin: The margin to use in loss calculation
        :param miner: The miner to use. If not provided, a default minuer that
            selects all possible triplets will be used
        """
        super().__init__()
        self.distance = distance
        self.margin = margin
        self.miner = miner

    def compute(
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
        if (
            ind_anch.nelement() == 0
            or ind_pos.nelement() == 0
            or ind_neg.nelement() == 0
        ):
            raise ValueError('Got empty tuple/triplets from your dataset.')

        dist_matrix = get_distance(embeddings, self.distance)
        dist_pos = dist_matrix[ind_anch, ind_pos]
        dist_neg = dist_matrix[ind_anch, ind_neg]
        loss = F.relu(dist_pos - dist_neg + self.margin)

        return loss.mean()

    def get_default_miner(
        self, is_session_dataset: bool
    ) -> Union[TripletMiner, TripletSessionMiner]:
        if not is_session_dataset:
            return TripletMiner()
        else:
            return TripletSessionMiner()


class NTXentLoss(nn.Module, BaseLoss[torch.Tensor]):
    """Compute the NTXent (Normalized Temeprature Cross-Entropy) loss.

    This loss function is a temperature-adjusted cross-entropy loss, as defined in the
    `SimCLR paper <https://arxiv.org/abs/2002.05709>`. It operates on batches where
    there are two views of each instance
    """

    def __init__(self, temperature: float = 0.1) -> None:
        """Initialize the loss instance.

        :param temerature: The temperature parameter
        """
        super().__init__()

        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        :param embeddings: An ``[N, d]`` tensor of embeddings.
        :param labels: An ``[N,]`` tensor of item labels. It is expected that each label
            appears two times.
        """
        assert embeddings.shape[0] == labels.shape[0]

        sim = (1 - get_distance(embeddings, 'cosine')) / self.temperature
        diag = torch.eye(sim.shape[0], dtype=sim.dtype, device=sim.device)
        labels1, labels2 = labels.unsqueeze(1), labels.unsqueeze(0)

        pos_samples = (labels1 == labels2).to(sim.dtype) - diag

        if not (pos_samples.sum(axis=1) == 1).all().item():
            raise ValueError('There need to be two views of each label in the batch.')

        self_mask = torch.ones_like(sim, requires_grad=False) - diag
        upper = torch.sum(sim * pos_samples, dim=1)
        lower = torch.log(torch.sum(self_mask * torch.exp(sim), dim=1))

        return -torch.mean(upper - lower)
