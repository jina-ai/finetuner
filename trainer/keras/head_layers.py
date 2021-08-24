import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..base import BaseHead


class HeadLayer(BaseHead, Layer):
    def call(self, *args, **kwargs):
        return self.forward(*args)


class CosineLayer(HeadLayer):
    arity = 2

    def get_output_for_loss(self, lvalue, rvalue):
        normalize_a = tf.nn.l2_normalize(lvalue, axis=-1)
        normalize_b = tf.nn.l2_normalize(rvalue, axis=-1)
        cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=-1)
        return cos_similarity

    def get_output_for_metric(self, lvalue, rvalue):
        return self.get_output_for_loss(lvalue, rvalue)

    def loss_fn(self, pred_val, target_val):
        return tf.keras.metrics.mean_squared_error(pred_val, target_val)

    def metric_fn(self, pred_val, target_val):
        s = tf.count_nonzero(
            tf.equal(tf.greater(pred_val, 0), tf.greater(target_val, 0))
        )
        return s / len(target_val)


class TripletLayer(HeadLayer):
    arity = 3

    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._margin = margin

    def get_output_for_loss(self, anchor, positive, negative):
        dist_pos = tf.norm(anchor - positive, ord='euclidean', axis=-1)
        dist_neg = tf.norm(anchor - negative, ord='euclidean', axis=-1)

        return tf.nn.relu(dist_pos - dist_neg + self._margin)

    def get_output_for_metric(self, anchor, positive, negative):
        normalize_a = tf.nn.l2_normalize(anchor, axis=-1)
        normalize_p = tf.nn.l2_normalize(positive, axis=-1)
        normalize_n = tf.nn.l2_normalize(negative, axis=-1)
        a_p_cos = tf.reduce_sum(tf.multiply(normalize_a, normalize_p), axis=-1)
        a_n_cos = tf.reduce_sum(tf.multiply(normalize_a, normalize_n), axis=-1)
        return a_p_cos, a_n_cos

    def loss_fn(self, pred_val, target_val):
        return tf.keras.metrics.mean_squared_error(target_val, pred_val)

    def metric_fn(self, pred_val, target_val):
        y_positive, y_negative = pred_val
        s_p = tf.count_nonzero(tf.greater(y_positive, 0))
        s_n = tf.count_nonzero(tf.less(y_negative, 0))
        return (s_p + s_n) / (len(y_positive) + len(y_negative))
