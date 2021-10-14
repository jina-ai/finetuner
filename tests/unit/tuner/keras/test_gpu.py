import pytest
import tensorflow as tf

from finetuner.tuner.keras import KerasTuner


@pytest.fixture(autouse=True)
def tf_gpu_config():
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=256)]
    )


@pytest.mark.gpu
@pytest.mark.parametrize('head_layer', ['TripletLayer', 'CosineLayer'])
def test_gpu_keras(generate_random_triplets, head_layer, caplog):
    data = generate_random_triplets(4, 4)
    embed_model = tf.keras.models.Sequential()
    embed_model.add(tf.keras.layers.InputLayer(input_shape=(4,)))
    embed_model.add(tf.keras.layers.Dense(4))

    tuner = KerasTuner(embed_model, head_layer)

    tuner.fit(data, data, epochs=2, batch_size=4, device='cuda')
