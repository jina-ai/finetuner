from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .miner import SiameseMiner, SiameseSessionMiner, TripletMiner, TripletSessionMiner
from ..base import BaseLoss, BaseMiner


def _is_tensor_empty(tensor: paddle.Tensor):
    return bool(tensor.is_empty())


def get_distance(embeddings: paddle.Tensor, distance: str) -> paddle.Tensor:
    """Get a matrix of pairwise distances between the embedings"""

    if distance == 'cosine':
        emb_norm = F.normalize(embeddings, p=2, axis=1)
        dists = 1 - paddle.mm(emb_norm, emb_norm.t())
    elif distance == 'euclidean':
        emb2 = (embeddings ** 2).sum(axis=1, keepdim=True)
        prod = paddle.mm(embeddings, embeddings.t())
        dists = emb2 + emb2.t() - 2 * prod
        dists = paddle.sqrt(dists.clip(0))
    elif distance == 'sqeuclidean':
        emb2 = (embeddings ** 2).sum(axis=1, keepdim=True)
        prod = paddle.mm(embeddings, embeddings.t())
        dists = emb2 + emb2.t() - 2 * prod

    return dists.clip(0)


class PaddleLoss(nn.Layer, BaseLoss[paddle.Tensor]):
    """Base class for all paddle losses."""

    def forward(
        self,
        embeddings: paddle.Tensor,
        labels: Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]],
    ) -> paddle.Tensor:
        if self.miner is None:
            # If labels is a tuple of tensors, this is a session dataset
            self.miner = self.get_default_miner(isinstance(labels, (list, tuple)))

        dists = get_distance(embeddings, self.distance)
        mined_tuples = self.miner.mine(labels, dists.clone().detach())
        loss = self.compute(embeddings, mined_tuples)

        return loss


class SiameseLoss(PaddleLoss):
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
        """
        super().__init__()
        self.distance = distance
        self.margin = margin
        self.miner = miner

    def compute(
        self,
        embeddings: paddle.Tensor,
        indices: Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor],
    ) -> paddle.Tensor:
        """Compute the loss

        :param embeddings: An ``[N, d]`` tensor of embeddings
        :param indices: A list of tuple indices and target, where each element in the
            list contains three elements: the indices of the two objects in the pair,
            and their similarity (which equals 1 if they are similar, and 0 if they
            are dissimilar)
        """
        ind_one, ind_two, target = indices
        if (
            _is_tensor_empty(ind_one)
            or _is_tensor_empty(ind_two)
            or _is_tensor_empty(target)
        ):
            raise ValueError('Got empty tuple/triplets from your dataset.')
        dist_matrix = get_distance(embeddings, self.distance)
        ind_slice = paddle.stack([ind_one, ind_two]).t()
        dists = paddle.gather_nd(dist_matrix, index=ind_slice)
        target = paddle.cast(target, "float32")

        loss = target * dists + (1 - target) * F.relu(self.margin - dists)
        return loss.mean()

    def get_default_miner(
        self, is_session_dataset: bool
    ) -> Union[SiameseMiner, SiameseSessionMiner]:
        if not is_session_dataset:
            return SiameseMiner()
        else:
            return SiameseSessionMiner()


class TripletLoss(PaddleLoss):
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
        distance: str = "cosine",
        margin: float = 1.0,
        miner: Optional[BaseMiner] = None,
    ):
        """Initialize the loss instance

        :param distance: The type of distance to use, avalilable options are
            ``"cosine"``, ``"euclidean"`` and ``"sqeuclidean"``
        :param margin: The margin to use in loss calculation
        """
        super().__init__()
        self.distance = distance
        self.margin = margin
        self.miner = miner

    def compute(
        self,
        embeddings: paddle.Tensor,
        indices: Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor],
    ) -> paddle.Tensor:
        """Compute the loss

        :param embeddings: An ``[N, d]`` tensor of embeddings
        :param indices: A list of tuple indices, where each element in the list
            contains three elements: the index of anchor, positive match and negative
            match in the embeddings tensor
        """
        ind_anch, ind_pos, ind_neg = indices
        if (
            _is_tensor_empty(ind_anch)
            or _is_tensor_empty(ind_pos)
            or _is_tensor_empty(ind_neg)
        ):
            raise ValueError('Got empty tuple/triplets from your dataset.')

        dist_matrix = get_distance(embeddings, self.distance)
        ind_slice_pos = paddle.stack([ind_anch, ind_pos]).t()
        ind_slice_neg = paddle.stack([ind_anch, ind_neg]).t()

        dist_pos = paddle.gather_nd(dist_matrix, index=ind_slice_pos)
        dist_neg = paddle.gather_nd(dist_matrix, index=ind_slice_neg)
        loss = F.relu(dist_pos - dist_neg + self.margin)

        return loss.mean()

    def get_default_miner(
        self, is_session_dataset: bool
    ) -> Union[SiameseMiner, SiameseSessionMiner]:
        if not is_session_dataset:
            return TripletMiner()
        else:
            return TripletSessionMiner()
