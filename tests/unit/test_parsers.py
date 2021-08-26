import paddle
import pytest
import tensorflow as tf
import torch

from trainer.paddle.parser import get_candidate_layers as gcl_p
from trainer.pytorch.parser import get_candidate_layers as gcl_t


@pytest.mark.parametrize(
    'user_model, parser',
    [
        (
            paddle.nn.Sequential(
                paddle.nn.Flatten(),
                paddle.nn.Linear(
                    in_features=28 * 28,
                    out_features=128,
                ),
                paddle.nn.ReLU(),
                paddle.nn.Linear(in_features=128, out_features=32),
            ),
            gcl_p,
        ),
        (
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(
                    in_features=28 * 28,
                    out_features=128,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=128, out_features=32),
            ),
            gcl_t,
        ),
    ],
)
def test_paddle_torch_model_parser(user_model, parser):
    r = parser(user_model, input_size=(1, 28, 28))
    assert len(r) == 4

    # flat layer can be a nonparametric candidate
    assert r[0]['output_features'] == 784
    assert r[0]['params'] == 0

    assert r[1]['output_features'] == 128
    assert r[1]['params'] == 100480

    # relu layer is a nonparametric candidate
    assert r[2]['output_features'] == 128
    assert r[2]['params'] == 0

    assert r[3]['output_features'] == 32
    assert r[3]['params'] == 4128


def test_keras_model_parser():
    user_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28), name='l1'),
            tf.keras.layers.Dense(128, activation='relu', name='l2'),
            tf.keras.layers.Dense(32, name='l3'),
        ]
    )

    from trainer.keras.parser import get_candidate_layers

    r = get_candidate_layers(user_model)
    assert len(r) == 3
    assert r[0]['name'] == 'l1'
    assert r[1]['name'] == 'l2'
    assert r[2]['name'] == 'l3'

    # flat layer can be a nonparametric candidate
    assert r[0]['output_features'] == 784
    assert r[0]['params'] == 0

    assert r[1]['output_features'] == 128
    assert r[1]['params'] == 100480

    assert r[2]['output_features'] == 32
    assert r[2]['params'] == 4128
