import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..base import BaseLoss


class CosineSiameseLoss(BaseLoss, Layer):
    """Computes the loss for a siamese network using cosine distance.

    The loss for a pair of objects equals ``(target - cos_sim)^2``, where ``target``
    should equal 1 when both objects belong to the same class, and to -1 when they
    belong to different classes. The ``cos_sim`` represents the cosime similarity
    between both objects.

    The final loss is the average over losses for all pairs of objects in the batch.
    """

    arity = 2

    def call(self, inputs, **kwargs):
        """Compute the loss.

        :param inputs: Should be a list or a tuple containing three tensors:
            - ``[N, D]`` tensor of embeddings of the first objects of the pair
            - ``[N, D]`` tensor of embeddings of the second objects of the pair
            - ``[N, ]`` tensor of target values
        """

        l_emb, r_emb, target = inputs
        normalize_a = tf.nn.l2_normalize(l_emb, axis=-1)
        normalize_b = tf.nn.l2_normalize(r_emb, axis=-1)
        cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=-1)
        return tf.keras.losses.mse(target, cos_similarity)


class EuclideanSiameseLoss(BaseLoss, Layer):
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

    def call(self, inputs, **kwargs):
        """Compute the loss.

        :param inputs: Should be a list or a tuple containing three tensors:
            - ``[N, D]`` tensor of embeddings of the first objects of the pair
            - ``[N, D]`` tensor of embeddings of the second objects of the pair
            - ``[N, ]`` tensor of target values
        """
        l_emb, r_emb, target = inputs
        eucl_dist = tf.reduce_sum(tf.math.squared_difference(l_emb, r_emb), axis=-1)
        is_similar = tf.cast(target > 0, tf.float32)

        loss = 0.5 * tf.square(
            is_similar * eucl_dist
            + (1 - is_similar) * tf.nn.relu(self.margin - eucl_dist)
        )
        return tf.reduce_mean(loss)


class EuclideanTripletLoss(BaseLoss, Layer):
    """Compute the loss for a triplet network using euclidean distance.

    The loss is computed as ``max(dist_pos - dist_neg + margin, 0)``, where ``dist_pos``
    is the euclidean distance between the anchor embedding and positive embedding,
    ``dist_neg`` is the euclidean distance between the anchor and negative embedding,
    and ``margin`` represents a wedge between the desired wedge between anchor-negative
    and anchor-positive distances.

    The final loss is the average over losses for all triplets in the batch.
    """

    arity = 3

    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._margin = margin

    def call(self, inputs, **kwargs):
        """Compute the loss.

        :param inputs: Should be a list or a tuple containing three tensors:
            - ``[N, D]`` tensor of embeddings of the anchor objects
            - ``[N, D]`` tensor of embeddings of the positive objects
            - ``[N, D]`` tensor of embeddings of the negative objects
        """
        anchor, positive, negative, _ = inputs

        # Seems that tf.norm suffers from numeric instability as explained here
        # https://github.com/tensorflow/tensorflow/issues/12071
        dist_pos = tf.reduce_sum(tf.math.squared_difference(anchor, positive), axis=-1)
        dist_neg = tf.reduce_sum(tf.math.squared_difference(anchor, negative), axis=-1)

        dist_pos = tf.sqrt(tf.maximum(dist_pos, 1e-9))
        dist_neg = tf.sqrt(tf.maximum(dist_neg, 1e-9))

        return tf.reduce_mean(tf.nn.relu(dist_pos - dist_neg + self._margin))


class CosineTripletLoss(BaseLoss, Layer):
    """Compute the loss for a triplet network using cosine distance.

    The loss is computed as ``max(dist_pos - dist_neg + margin, 0)``, where ``dist_pos``
    is the cosine distance between the anchor embedding and positive embedding,
    ``dist_neg`` is the cosine distance between the anchor and negative embedding, and
    ``margin`` represents a wedge between the desired wedge between anchor-negative and
    anchor-positive distances.

    The final loss is the average over losses for all triplets in the batch.
    """

    arity = 3

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self._margin = margin

    def call(self, inputs, **kwargs):
        """Compute the loss.

        :param inputs: Should be a list or a tuple containing three tensors:
            - ``[N, D]`` tensor of embeddings of the anchor objects
            - ``[N, D]`` tensor of embeddings of the positive objects
            - ``[N, D]`` tensor of embeddings of the negative objects
        """
        anchor, positive, negative, _ = inputs

        # Seems that tf.norm suffers from numeric instability as explained here
        # https://github.com/tensorflow/tensorflow/issues/12071
        normalize_a = tf.nn.l2_normalize(anchor, axis=-1)
        normalize_p = tf.nn.l2_normalize(positive, axis=-1)
        normalize_n = tf.nn.l2_normalize(negative, axis=-1)
        dist_pos = 1 - tf.reduce_sum(tf.multiply(normalize_a, normalize_p), axis=-1)
        dist_neg = 1 - tf.reduce_sum(tf.multiply(normalize_a, normalize_n), axis=-1)

        return tf.reduce_mean(tf.nn.relu(dist_pos - dist_neg + self._margin))
