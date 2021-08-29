import json

import paddle
import tensorflow as tf
import torch

import finetuner
from tests.data_generator import fashion_match_doc_generator as mdg


def test_fit_all(tmpdir):
    base_models = {
        'keras': tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(32),
            ]
        ),
        'pytorch': torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=28 * 28,
                out_features=128,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=32),
        ),
        'paddle': paddle.nn.Sequential(
            paddle.nn.Flatten(),
            paddle.nn.Linear(
                in_features=28 * 28,
                out_features=128,
            ),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=128, out_features=32),
        ),
    }

    for kb, b in base_models.items():
        for h in ['CosineLayer', 'TripletLayer']:
            result = finetuner.fit(
                b,
                head_layer=h,
                train_data=mdg(num_total=300),
                eval_data=lambda: mdg(num_total=300, is_testset=True),
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
