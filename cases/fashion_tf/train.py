# Train a fashion classifier based on TF tutorial: https://www.tensorflow.org/tutorials/keras/classification
# no surprise here

import tensorflow as tf

from data_gen import download_data, targets

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# download the data
download_data(targets)

model.fit(targets['index']['data'], targets['index-labels']['data'], epochs=10)
