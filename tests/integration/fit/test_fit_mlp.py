import paddle
import tensorflow as tf
import torch

import finetuner
from finetuner.toydata import generate_fashion_match_catalog

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
            train_data, train_catalog = generate_fashion_match_catalog(
                num_neg=10,
                num_pos=10,
                num_total=300,
                num_catalog=3000,
                pre_init_generator=False,
            )
            eval_data, eval_catalog = generate_fashion_match_catalog(
                num_neg=10,
                num_pos=10,
                num_total=300,
                num_catalog=3000,
                is_testset=True,
                pre_init_generator=False,
            )
            train_catalog.extend(eval_catalog)
            result = finetuner.fit(
                b(),
                loss=h,
                train_data=train_data,
                eval_data=eval_data,
                catalog=train_catalog,
                epochs=2,
            )
            result.save(tmpdir / f'result-{kb}-{h}.json')
