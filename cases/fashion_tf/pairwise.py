# Train a fashion classifier based on TF tutorial: https://www.tensorflow.org/tutorials/keras/classification
# no surprise here

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers

from data_gen import download_data, targets

target_shape = (28, 28)
user_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=target_shape),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


# trainable = False
# for layer in model.layers:
#     print(layer.name)
#     #
#     # if layer.name == "conv5_block1_out":
#     #     trainable = True
#     # layer.trainable = trainable


class HatLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def call(self, lvalue, rvalue):
        return tf.concat([lvalue, rvalue, tf.abs(lvalue - rvalue)], axis=-1)


lvalue = keras.Input(shape=target_shape)
rvalue = keras.Input(shape=target_shape)

hat_layer = HatLayer()(user_model(lvalue),
                                 user_model(rvalue))

wrapped_model = Model(
    inputs=[lvalue, rvalue], outputs=hat_layer
)

wrapped_model.summary()

exit()
user_model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

# download the data
download_data(targets)

user_model.fit(targets['index']['data'], targets['index-labels']['data'], epochs=10)
