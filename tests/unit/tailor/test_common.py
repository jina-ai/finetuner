import paddle
import pytest
import tensorflow as tf
import torch

from finetuner.helper import get_framework
from finetuner.tailor import to_embedding_model, display


class LastCellPT(torch.nn.Module):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


class LastCellPD(paddle.nn.Layer):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


embed_models = {
    'keras': lambda: tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(32),
        ]
    ),
    'torch': lambda: torch.nn.Sequential(
        torch.nn.Embedding(num_embeddings=5000, embedding_dim=64),
        torch.nn.LSTM(64, 64, bidirectional=True, batch_first=True),
        LastCellPT(),
        torch.nn.Linear(in_features=2 * 64, out_features=32),
    ),
    'paddle': lambda: paddle.nn.Sequential(
        paddle.nn.Embedding(num_embeddings=5000, embedding_dim=64),
        paddle.nn.LSTM(64, 64, direction='bidirectional'),
        LastCellPD(),
        paddle.nn.Linear(in_features=2 * 64, out_features=32),
    ),
}


@pytest.mark.parametrize('framework', ['keras', 'paddle', 'torch'])
@pytest.mark.parametrize('freeze', [True, False])
@pytest.mark.parametrize('output_dim', [None, 2])
def test_to_embedding_fn(framework, output_dim, freeze):
    m = embed_models[framework]()
    assert get_framework(m) == framework
    m1 = to_embedding_model(
        m, input_size=(5000,), input_dtype='int64', freeze=freeze, output_dim=output_dim
    )
    assert m1
    assert get_framework(m1) == framework


@pytest.mark.parametrize('framework', ['keras', 'paddle', 'torch'])
def test_display(framework):
    m = embed_models[framework]()
    display(m, input_size=(5000,), input_dtype='int64')
