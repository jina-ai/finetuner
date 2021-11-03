import pytest
import tensorflow as tf


@pytest.fixture(autouse=True)
def clear_session():
    tf.keras.backend.clear_session()
