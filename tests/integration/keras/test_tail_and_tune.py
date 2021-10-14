import pytest
import tensorflow as tf

from finetuner import fit


@pytest.fixture
def embed_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(128,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32),
        ]
    )


def test_tail_and_tune(embed_model, create_easy_data):
    data, _ = create_easy_data(10, 128, 1000)
    rv = fit(
        model=embed_model,
        train_data=data,
        epochs=5,
        to_embedding_model=True,
        input_size=128,
        output_dim=16,
        layer_name='dense_2',
    )
    assert rv['loss']['train']
    assert rv['metric']['train']
