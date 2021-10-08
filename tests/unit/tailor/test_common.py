import paddle
import pytest
import tensorflow as tf
import torch

from finetuner.tailor import to_embedding_model


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
    'pytorch': lambda: torch.nn.Sequential(
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


@pytest.mark.parametrize('framework', ['keras', 'pytorch', 'paddle'])
def test_to_embedding_fn(framework):
    m = embed_models[framework]()
    m1 = to_embedding_model(m)
