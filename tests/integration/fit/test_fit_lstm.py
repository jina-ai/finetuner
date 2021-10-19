import paddle
import tensorflow as tf
import torch

from finetuner import fit
from finetuner.toydata import generate_qa_match_catalog

all_test_losses = [
    'CosineSiameseLoss',
    'CosineTripletLoss',
    'EuclideanSiameseLoss',
    'EuclideanTripletLoss',
]


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
        for h in all_test_losses:
            train_data, train_catalog = generate_qa_match_catalog(
                num_total=300, num_neg=5, max_seq_len=10, pre_init_generator=False
            )
            eval_data, eval_catalog = generate_qa_match_catalog(
                num_total=300, num_neg=5, max_seq_len=10, pre_init_generator=False
            )
            train_catalog.extend(eval_catalog)
            result = fit(
                b(),
                loss=h,
                train_data=train_data,
                eval_data=eval_data,
                catalog=train_catalog,
                epochs=2,
            )
            result.save(tmpdir / f'result-{kb}-{h}.json')
