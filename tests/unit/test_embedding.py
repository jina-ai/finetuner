import paddle
import pytest
import tensorflow as tf
import torch
from jina import DocumentArray, DocumentArrayMemmap, Document
import numpy as np

from finetuner.embedding import embed
from finetuner.toydata import generate_fashion

embed_models = {
    "keras": lambda: tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(32),
        ]
    ),
    "pytorch": lambda: torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=32),
    ),
    "paddle": lambda: paddle.nn.Sequential(
        paddle.nn.Flatten(),
        paddle.nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        paddle.nn.ReLU(),
        paddle.nn.Linear(in_features=128, out_features=32),
    ),
}

random_embed_models = {
    "keras": lambda: tf.keras.Sequential(
        [tf.keras.layers.Dropout(0.5), tf.keras.layers.BatchNormalization()]
    ),
    "pytorch": lambda: torch.nn.Sequential(
        torch.nn.Dropout(0.5), torch.nn.BatchNorm1d(128)
    ),
    "paddle": lambda: paddle.nn.Sequential(
        paddle.nn.Dropout(0.5), paddle.nn.BatchNorm1D(128)
    ),
}


@pytest.mark.parametrize("framework", ["keras", "pytorch", "paddle"])
def test_embedding_on_random_network(framework):
    docs = DocumentArray([Document() for _ in range(2)])
    docs.blobs = np.random.random([2, 128]).astype(np.float32)
    embed_model = random_embed_models[framework]()
    embed(docs, embed_model)

    embed1 = docs.embeddings.copy()

    # reset
    docs.embeddings = np.random.random([2, 128]).astype(np.float32)

    # try it again, it should yield the same result
    embed(docs, embed_model)
    np.testing.assert_array_almost_equal(docs.embeddings, embed1)

    # reset
    docs.embeddings = np.random.random([2, 128]).astype(np.float32)

    # now do this one by one
    embed(docs[:1], embed_model)
    embed(docs[-1:], embed_model)
    np.testing.assert_array_almost_equal(docs.embeddings, embed1)


@pytest.mark.parametrize("framework", ["keras", "pytorch", "paddle"])
def test_set_embeddings(framework, tmpdir):
    # works for DA
    embed_model = embed_models[framework]()
    docs = DocumentArray(generate_fashion(num_total=100))
    embed(docs, embed_model)
    assert docs.embeddings.shape == (100, 32)

    # works for DAM
    dam = DocumentArrayMemmap(tmpdir)
    dam.extend(generate_fashion(num_total=42))
    embed(dam, embed_model)
    assert dam.embeddings.shape == (42, 32)
