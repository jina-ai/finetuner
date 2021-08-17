import abc

import tensorflow as tf
from tensorflow.keras.layers import Layer


class HeadLayer(Layer):
    default_loss: str  #: the recommended loss function to be used when equipping this layer to base model

    @abc.abstractmethod
    def call(self, inputs, **kwargs):
        ...


class PairwiseHeadLayer(HeadLayer):
    @abc.abstractmethod
    def call(self, lvalue, rvalue, **kwargs):
        ...


class HatLayer(PairwiseHeadLayer):
    default_loss = 'hinge'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(1)

    def call(self, lvalue, rvalue):
        x = tf.concat([lvalue, rvalue, tf.abs(lvalue - rvalue)], axis=-1)
        return self.fc(x)


class DistanceLayer(PairwiseHeadLayer):
    default_loss = 'hinge'

    def call(self, lvalue, rvalue):
        return -tf.reduce_sum(
            tf.squared_difference(lvalue, rvalue), axis=-1, keepdims=True
        )


class DiffLayer(PairwiseHeadLayer):
    default_loss = 'mse'

    def call(self, lvalue, rvalue):
        return tf.reduce_sum(tf.abs(lvalue - rvalue), axis=-1, keepdims=True)


class CosineLayer(PairwiseHeadLayer):

    default_loss = 'mse'

    def call(self, lvalue, rvalue):
        normalize_a = tf.nn.l2_normalize(lvalue, axis=-1)
        normalize_b = tf.nn.l2_normalize(rvalue, axis=-1)
        cos_similarity = tf.reduce_sum(
            tf.multiply(normalize_a, normalize_b), axis=-1, keepdims=True
        )
        return cos_similarity
