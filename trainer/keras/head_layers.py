import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..base import BaseHead


class HeadLayer(BaseHead, Layer):
    def call(self, *args, **kwargs):
        return self.forward(*args)


class CosineLayer(HeadLayer):
    arity = 2

    def get_output(self, lvalue, rvalue):
        normalize_a = tf.nn.l2_normalize(lvalue, axis=-1)
        normalize_b = tf.nn.l2_normalize(rvalue, axis=-1)
        cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=-1)
        return cos_similarity

    def loss_fn(self, pred_val, target_val):
        return tf.keras.losses.mse(target_val, pred_val)

    def metric_fn(self, pred_val, target_val):
        s = tf.math.count_nonzero(
            tf.equal(tf.greater(pred_val, 0), tf.greater(target_val, 0))
        )
        return s / len(target_val)


class TripletLayer(HeadLayer):
    arity = 3

    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._margin = margin

    def get_output(self, anchor, positive, negative):
        # Seems that tf.norm suffers from numeric instability as explained here
        # https://github.com/tensorflow/tensorflow/issues/12071
        dist_pos = tf.reduce_sum(tf.math.squared_difference(anchor, positive), axis=-1)
        dist_neg = tf.reduce_sum(tf.math.squared_difference(anchor, negative), axis=-1)

        return dist_pos, dist_neg

    def loss_fn(self, pred_val, target_val):
        dist_pos, dist_neg = pred_val
        return tf.reduce_mean(tf.nn.relu(dist_pos - dist_neg + self._margin))

    def metric_fn(self, pred_val, target_val):
        dist_pos, dist_neg = pred_val
        s = tf.math.count_nonzero(tf.less(dist_pos, dist_neg))
        return s / len(target_val)
