import tensorflow as tf
from paddle import nn


def test_paddle_model_parser():
    user_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=32),
    )

    from trainer.paddle.parser import get_candidate_layers

    r = get_candidate_layers(user_model, input_size=(1, 28, 28))
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
