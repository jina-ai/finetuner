import json

import paddle
import tensorflow as tf
import torch

import finetuner
from finetuner.toydata import generate_fashion_match

all_test_losses = [
    'CosineSiameseLoss',
    'CosineTripletLoss',
    'EuclideanSiameseLoss',
    'EuclideanTripletLoss',
]


def test_fit_all(tmpdir):
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

    for kb, b in embed_models.items():
        for h in all_test_losses:
            result = finetuner.fit(
                b(),
                loss=h,
                train_data=lambda: generate_fashion_match(
                    num_neg=10, num_pos=10, num_total=300
                ),
                eval_data=lambda: generate_fashion_match(
                    num_neg=10, num_pos=10, num_total=300, is_testset=True
                ),
                epochs=2,
            )
            result.save(tmpdir / f'result-{kb}-{h}.json')
