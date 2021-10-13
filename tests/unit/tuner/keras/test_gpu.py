import pytest
import tensorflow as tf

from finetuner.tuner.keras import KerasTuner


# @pytest.mark.gpu
@pytest.mark.parametrize('head_layer', ['TripletLayer', 'CosineLayer'])
def test_gpu_keras(generate_random_triplets, head_layer, caplog):
    tf.debugging.set_log_device_placement(True)
    data = generate_random_triplets(4, 4)
    embed_model = tf.keras.models.Sequential()
    embed_model.add(tf.keras.layers.InputLayer(input_shape=(4,)))  # (None, 128)
    embed_model.add(tf.keras.layers.Dense(4))  # (None, 128)

    tuner = KerasTuner(embed_model, head_layer)

    tuner.fit(data, data, epochs=2, batch_size=4, device='cuda')
