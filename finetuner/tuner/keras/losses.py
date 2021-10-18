import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..base import BaseLoss


class CosineSiameseLoss(BaseLoss, Layer):
    arity = 2

    def call(self, inputs, **kwargs):
        l_emb, r_emb, target = inputs
        normalize_a = tf.nn.l2_normalize(l_emb, axis=-1)
        normalize_b = tf.nn.l2_normalize(r_emb, axis=-1)
        cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=-1)
        return tf.keras.losses.mse(target, cos_similarity)


class EuclideanSiameseLoss(BaseLoss, Layer):
    arity = 2

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def call(self, inputs, **kwargs):
        l_emb, r_emb, target = inputs
        eucl_dist = tf.reduce_sum(tf.math.squared_difference(l_emb, r_emb), axis=-1)
        is_similar = tf.cast(target > 0, tf.float32)

        loss = 0.5 * tf.square(
            is_similar * eucl_dist
            + (1 - is_similar) * tf.nn.relu(self.margin - eucl_dist)
        )
        return tf.reduce_mean(loss)


class EuclideanTripletLoss(BaseLoss, Layer):
    arity = 3

    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._margin = margin

    def call(self, inputs, **kwargs):
        anchor, positive, negative, _ = inputs

        # Seems that tf.norm suffers from numeric instability as explained here
        # https://github.com/tensorflow/tensorflow/issues/12071
        dist_pos = tf.reduce_sum(tf.math.squared_difference(anchor, positive), axis=-1)
        dist_neg = tf.reduce_sum(tf.math.squared_difference(anchor, negative), axis=-1)

        dist_pos = tf.maximum(dist_pos, 1e-9)
        dist_neg = tf.maximum(dist_neg, 1e-9)

        return tf.reduce_mean(
            tf.nn.relu(tf.sqrt(dist_pos) - tf.sqrt(dist_neg) + self._margin)
        )


class CosineTripletLoss(BaseLoss, Layer):
    arity = 3

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self._margin = margin

    def call(self, inputs, **kwargs):
        anchor, positive, negative, _ = inputs

        # Seems that tf.norm suffers from numeric instability as explained here
        # https://github.com/tensorflow/tensorflow/issues/12071
        normalize_a = tf.nn.l2_normalize(anchor, axis=-1)
        normalize_p = tf.nn.l2_normalize(positive, axis=-1)
        normalize_n = tf.nn.l2_normalize(negative, axis=-1)
        dist_pos = 1 - tf.reduce_sum(tf.multiply(normalize_a, normalize_p), axis=-1)
        dist_neg = 1 - tf.reduce_sum(tf.multiply(normalize_a, normalize_n), axis=-1)

        return tf.reduce_mean(tf.nn.relu(dist_pos - dist_neg + self._margin))
