import paddle
import pytest
import tensorflow as tf
import torch
from jina import DocumentArray

from finetuner.embedding import fill_embeddings
from finetuner.toydata import generate_fashion_match

embed_models = {
    'keras': lambda: tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32),
        ]
    ),
    'pytorch': lambda: torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=32),
    ),
    'paddle': lambda: paddle.nn.Sequential(
        paddle.nn.Flatten(),
        paddle.nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        paddle.nn.ReLU(),
        paddle.nn.Linear(in_features=128, out_features=32),
    ),
}


@pytest.mark.parametrize('framework', ['keras', 'pytorch', 'paddle'])
def test_embedding_docs(framework):
    embed_model = embed_models[framework]
    docs = DocumentArray(generate_fashion_match(num_total=100))
    fill_embeddings(docs, embed_model)
