from typing import Optional, Tuple, Union

import tensorflow as tf

from .miner import SiameseMiner, SiameseSessionMiner, TripletMiner, TripletSessionMiner
from ..base import BaseLoss, BaseMiner


def _is_tensor_empty(tensor: tf.Tensor):
    return bool(tf.equal(tf.size(tensor), 0))


def get_distance(embeddings: tf.Tensor, distance: str) -> tf.Tensor:
    """Get a matrix of pairwise distances between the embeddings"""

    if distance == 'cosine':
        emb_norm, _ = tf.linalg.normalize(embeddings, ord=2, axis=1)
        dists = 1 - tf.linalg.matmul(emb_norm, tf.transpose(emb_norm))
    elif distance == 'euclidean':
        embed = tf.reduce_sum(embeddings ** 2, axis=1, keepdims=True)
        prod = tf.linalg.matmul(embeddings, tf.transpose(embeddings))
        dists = embed + tf.transpose(embed) - 2 * prod
        dists = tf.sqrt(
            tf.clip_by_value(dists, clip_value_min=0, clip_value_max=tf.float64.max)
        )
    elif distance == 'sqeuclidean':
        embed = tf.reduce_sum(embeddings ** 2, axis=1, keepdims=True)
        prod = tf.linalg.matmul(embeddings, tf.transpose(embeddings))
        dists = embed + tf.transpose(embed) - 2 * prod

    return tf.clip_by_value(dists, clip_value_min=0, clip_value_max=tf.float64.max)


class KerasLoss(tf.keras.layers.Layer, BaseLoss[tf.Tensor]):
    """Base class for all keras/tensorflow losses."""

    def call(
        self,
        embeddings: tf.Tensor,
        labels: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
    ) -> tf.Tensor:
        if self.miner is None:
            # If labels is a tuple of tensors, this is a session dataset
            self.miner = self.get_default_miner(isinstance(labels, (list, tuple)))

        dists = get_distance(embeddings, self.distance)
        mined_tuples = self.miner.mine(labels, tf.identity(dists))
        loss = self.compute(embeddings, mined_tuples)

        return loss


class SiameseLoss(KerasLoss):
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
        embeddings: tf.Tensor,
        indices: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tf.Tensor:
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
        ind_slice = tf.transpose([ind_one, ind_two])
        dists = tf.gather_nd(dist_matrix, indices=[ind_slice])
        target = tf.cast(target, tf.float32)

        loss = target * dists + (1 - target) * tf.nn.relu(self.margin - dists)
        return tf.math.reduce_mean(loss, axis=1)

    def get_default_miner(
        self, is_session_dataset: bool
    ) -> Union[SiameseMiner, SiameseSessionMiner]:
        if not is_session_dataset:
            return SiameseMiner()
        else:
            return SiameseSessionMiner()


class TripletLoss(KerasLoss):
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
        :param miner: The miner to use. If not provided, a default minuer that
            selects all possible triplets will be used
        """
        super().__init__()
        self.distance = distance
        self.margin = margin
        self.miner = miner

    def compute(
        self,
        embeddings: tf.Tensor,
        indices: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tf.Tensor:
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
        ind_slice_pos = tf.transpose([ind_anch, ind_pos])
        ind_slice_neg = tf.transpose([ind_anch, ind_neg])

        dist_pos = tf.gather_nd(dist_matrix, indices=[ind_slice_pos])
        dist_neg = tf.gather_nd(dist_matrix, indices=[ind_slice_neg])
        loss = tf.nn.relu(dist_pos - dist_neg + self.margin)

        return tf.math.reduce_mean(loss, axis=1)

    def get_default_miner(
        self, is_session_dataset: bool
    ) -> Union[TripletMiner, TripletSessionMiner]:
        if not is_session_dataset:
            return TripletMiner()
        else:
            return TripletSessionMiner()
