import tensorflow as tf
from tensorflow.keras.layers import Layer


class HatLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(1)

    def call(self, lvalue, rvalue):
        x = tf.concat([lvalue, rvalue, tf.abs(lvalue - rvalue)], axis=-1)
        return self.fc(x)


class CosineLayer(Layer):

    def call(self, lvalue, rvalue):
        normalize_a = tf.nn.l2_normalize(lvalue)
        normalize_b = tf.nn.l2_normalize(rvalue)
        cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=-1, keepdims=True)
        return cos_similarity
