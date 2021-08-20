import abc

import tensorflow as tf
from tensorflow.keras.layers import Layer


class HeadLayer(Layer):
    default_loss: str  #: the recommended loss function to be used when equipping this layer to base model
    arity: int  #: the arity of the inputs

    @abc.abstractmethod
    def call(self, *args, **kwargs):
        ...


class HatLayer(HeadLayer):
    default_loss = 'hinge'
    arity = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(1)

    def call(self, lvalue, rvalue):
        x = tf.concat([lvalue, rvalue, tf.abs(lvalue - rvalue)], axis=-1)
        return self.fc(x)


class DistanceLayer(HeadLayer):
    default_loss = 'hinge'
    arity = 2

    def call(self, lvalue, rvalue):
        return -tf.reduce_sum(
            tf.squared_difference(lvalue, rvalue), axis=-1, keepdims=True
        )


class DiffLayer(HeadLayer):
    default_loss = 'mse'
    arity = 2

    def call(self, lvalue, rvalue):
        return tf.reduce_sum(tf.abs(lvalue - rvalue), axis=-1, keepdims=True)


class CosineLayer(HeadLayer):
    default_loss = 'mse'
    arity = 2

    def call(self, lvalue, rvalue):
        normalize_a = tf.nn.l2_normalize(lvalue, axis=-1)
        normalize_b = tf.nn.l2_normalize(rvalue, axis=-1)
        cos_similarity = tf.reduce_sum(
            tf.multiply(normalize_a, normalize_b), axis=-1, keepdims=True
        )
        return cos_similarity


class TripletLayer(HeadLayer):
    default_loss = 'mse'
    arity = 3

    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._margin = margin

    def call(self, anchor, positive, negative):
        dist_pos = tf.norm(anchor - positive, ord='euclidean', axis=-1, keepdims=True)
        dist_neg = tf.norm(anchor - negative, ord='euclidean', axis=-1, keepdims=True)

        return tf.nn.relu(dist_pos - dist_neg + self._margin)
