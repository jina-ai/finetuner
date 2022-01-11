from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..base import BaseLoss, BaseMiner
from .miner import SiameseMiner, SiameseSessionMiner, TripletMiner, TripletSessionMiner


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


class PaddleTupleLoss(nn.Layer, BaseLoss[paddle.Tensor]):
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


class SiameseLoss(PaddleTupleLoss):
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
        target = paddle.cast(target, 'float32')

        loss = target * dists + (1 - target) * F.relu(self.margin - dists)
        return loss.mean()

    def get_default_miner(
        self, is_session_dataset: bool
    ) -> Union[SiameseMiner, SiameseSessionMiner]:
        if not is_session_dataset:
            return SiameseMiner()
        else:
            return SiameseSessionMiner()


class TripletLoss(PaddleTupleLoss):
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


class NTXentLoss(nn.Layer, BaseLoss[paddle.Tensor]):
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

    def forward(
        self, embeddings: paddle.Tensor, labels: paddle.Tensor
    ) -> paddle.Tensor:
        """Compute the loss.

        :param embeddings: An ``[N, d]`` tensor of embeddings.
        :param labels: An ``[N,]`` tensor of item labels. It is expected that each label
            appears two times.
        """
        assert embeddings.shape[0] == labels.shape[0]

        sim = (1 - get_distance(embeddings, 'cosine')) / self.temperature
        diag = paddle.eye(sim.shape[0], dtype=sim.dtype)
        diag = paddle.to_tensor(diag, place=sim.place)
        labels1, labels2 = labels.unsqueeze(1), labels.unsqueeze(0)

        pos_samples = paddle.cast(labels1 == labels2, sim.dtype) - diag

        if not (pos_samples.sum(axis=1) == 1).all().item():
            raise ValueError('There need to be two views of each label in the batch.')

        self_mask = paddle.ones_like(sim) - diag
        upper = paddle.sum(sim * pos_samples, axis=1)
        lower = paddle.log(paddle.sum(self_mask * paddle.exp(sim), axis=1))

        return -paddle.mean(upper - lower)
