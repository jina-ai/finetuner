import json

import paddle
import tensorflow as tf
import torch

from finetuner import fit
from finetuner.toydata import generate_qa_match


class LastCellPT(torch.nn.Module):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


class LastCellPD(paddle.nn.Layer):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


def test_fit_all(tmpdir):
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

    for kb, b in embed_models.items():
        for h in ['CosineLayer', 'TripletLayer']:
            result = fit(
                b(),
                head_layer=h,
                train_data=lambda: generate_qa_match(
                    num_total=300, num_neg=5, max_seq_len=10
                ),
                eval_data=lambda: generate_qa_match(
                    num_total=300, num_neg=5, max_seq_len=10
                ),
                epochs=2,
            )

            # convert from numpy to python native float for json dump
            result = {
                'loss': {
                    'train': [float(v) for v in result['loss']['train']],
                    'eval': [float(v) for v in result['loss']['eval']],
                },
                'metric': {
                    'train': [float(v) for v in result['metric']['train']],
                    'eval': [float(v) for v in result['metric']['eval']],
                },
            }
            with open(tmpdir / f'result-{kb}-{h}.json', 'w') as fp:
                json.dump(result, fp)
