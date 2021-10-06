import json

import paddle
import tensorflow as tf
import torch

import finetuner
from finetuner.toydata import generate_fashion_match


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
        for h in ['CosineLayer', 'TripletLayer']:
            result = finetuner.fit(
                b(),
                head_layer=h,
                train_data=lambda: generate_fashion_match(
                    num_neg=10, num_pos=10, num_total=300
                ),
                eval_data=lambda: generate_fashion_match(
                    num_neg=10, num_pos=10, num_total=300, is_testset=True
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
