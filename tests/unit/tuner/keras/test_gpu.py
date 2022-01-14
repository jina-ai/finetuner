import pytest
import tensorflow as tf
from docarray import DocumentArray

from finetuner.embedding import embed
from finetuner.toydata import generate_fashion
from finetuner.tuner.keras import KerasTuner

all_test_losses = ['SiameseLoss', 'TripletLoss']


@pytest.mark.gpu
@pytest.mark.parametrize('loss', all_test_losses)
def test_gpu_keras(generate_random_data, loss, tf_gpu_config):
    data = generate_random_data(40, 4)
    embed_model = tf.keras.models.Sequential()
    embed_model.add(tf.keras.layers.InputLayer(input_shape=(4,)))
    embed_model.add(tf.keras.layers.Dense(4))

    tuner = KerasTuner(embed_model, loss, device='cuda')

    tuner.fit(data, data, epochs=2, batch_size=8)


@pytest.mark.gpu
def test_set_embeddings_gpu(tmpdir, tf_gpu_config):
    # works for DA
    embed_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32),
        ]
    )
    docs = DocumentArray(generate_fashion(num_total=100))
    embed(docs, embed_model, 'cuda')
    assert docs.embeddings.shape == (100, 32)