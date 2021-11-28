import pytest
import tensorflow as tf


@pytest.fixture
def tf_gpu_config():
    tf.keras.backend.clear_session()
    return
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=256)]
    )
