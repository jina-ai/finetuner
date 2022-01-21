import tensorflow as tf


class ProjectionHead(tf.keras.layers.Layer):
    """Projection head used internally for self-supervised training.
    It is (by default) a simple 3-layer MLP to be attached on top of embedding model only for training purpose.
    After training, it should be cut-out from the embedding model.
    """

    EPSILON = 1e-5

    def __init__(self, in_features: int, output_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.layers = []
        for idx in range(num_layers - 1):
            self.layers.append(
                tf.keras.layers.Dense(
                    units=in_features,
                    bias_initializer='zeros',
                )
            )
            self.layers.append(tf.keras.layers.BatchNormalization(epsilon=self.EPSILON))
            self.layers.append(tf.keras.layers.ReLU())
        self.layers.append(
            tf.keras.layers.Dense(
                units=output_dim,
                bias_initializer='zeros',
            )
        )
        self.layers.append(tf.keras.layers.BatchNormalization(epsilon=self.EPSILON))

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
