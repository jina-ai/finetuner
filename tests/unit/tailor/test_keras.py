import pytest
import tensorflow as tf
import numpy as np

from finetuner.tailor.keras import KerasTailor


@pytest.fixture
def dense_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(128,)))  # (None, 128)
    model.add(tf.keras.layers.Dense(128, activation='relu'))  # (None, 128)
    model.add(tf.keras.layers.Dense(64, activation='relu'))  # (None, 64)
    model.add(tf.keras.layers.Dense(32, activation='relu'))  # (None, 32)
    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # (None, 10)
    return model


@pytest.fixture
def simple_cnn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(32, 3, (1, 1), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, 3, (1, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


@pytest.fixture
def vgg16_cnn_model():
    return tf.keras.applications.vgg16.VGG16()


@pytest.fixture
def stacked_lstm():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(1000, 1024, input_length=128))
    model.add(
        tf.keras.layers.LSTM(256, return_sequences=True)
    )  # this layer will not considered as candidate layer
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(
        tf.keras.layers.LSTM(256, return_sequences=False)
    )  # this layer will be considered as candidate layer
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    return model


@pytest.fixture
def bidirectional_lstm():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(32),
        ]
    )


@pytest.fixture(
    params=[
        'dense_model',
        'simple_cnn_model',
        'vgg16_cnn_model',
        'stacked_lstm',
        'bidirectional_lstm',
    ]
)
def model(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    'model, layer_name',
    [
        ('dense_model', 'random_name'),
        ('simple_cnn_model', 'random_name'),
        ('vgg16_cnn_model', 'random_name'),
        ('stacked_lstm', 'random_name'),
        ('bidirectional_lstm', 'random_name'),
    ],
    indirect=['model'],
)
def test_trim_fail_given_unexpected_layer_name(model, layer_name):
    with pytest.raises(KeyError):
        keras_tailor = KerasTailor(model)
        keras_tailor.to_embedding_model(layer_name=layer_name)


@pytest.mark.parametrize(
    'model, layer_name, expected_output_shape',
    [
        ('dense_model', 'dense_3', (None, 10)),
        ('simple_cnn_model', 'dense', (None, 128)),
        ('vgg16_cnn_model', 'fc2', (None, 4096)),
        ('stacked_lstm', 'dense', (None, 256)),
        ('bidirectional_lstm', 'dense', (None, 32)),
        ('dense_model', None, (None, 10)),
        ('simple_cnn_model', None, (None, 10)),
        ('vgg16_cnn_model', None, (None, 1000)),
        ('stacked_lstm', None, (None, 5)),
        ('bidirectional_lstm', None, (None, 32)),
    ],
    indirect=['model'],
)
def test_to_embedding_model(model, layer_name, expected_output_shape):
    keras_tailor = KerasTailor(model)
    model = keras_tailor.to_embedding_model(layer_name=layer_name)
    assert model.output_shape == expected_output_shape


def test_weights_preserved_given_pretrained_model(vgg16_cnn_model):
    weights = vgg16_cnn_model.layers[0].get_weights()
    keras_tailor = KerasTailor(vgg16_cnn_model)
    vgg16_cnn_model = keras_tailor.to_embedding_model(layer_name='fc2')
    weights_after_convert = vgg16_cnn_model.layers[0].get_weights()
    np.testing.assert_array_equal(weights, weights_after_convert)


@pytest.mark.parametrize(
    'model, layer_name, output_dim, expected_output_shape',
    [
        ('dense_model', 'dense_3', None, (None, 10)),
        ('simple_cnn_model', 'dense', None, (None, 128)),
        ('vgg16_cnn_model', 'fc2', None, (None, 4096)),
        ('stacked_lstm', 'dense', None, (None, 256)),
        ('bidirectional_lstm', 'dense', None, (None, 32)),
        # no layer name no output dim
        ('dense_model', None, None, (None, 10)),
        ('simple_cnn_model', None, None, (None, 10)),
        ('vgg16_cnn_model', None, None, (None, 1000)),
        ('stacked_lstm', None, None, (None, 5)),
        ('bidirectional_lstm', None, None, (None, 32)),
        # with output dim
        ('dense_model', 'dense_3', 16, (None, 16)),
        ('simple_cnn_model', 'dense', 1024, (None, 1024)),
        ('vgg16_cnn_model', 'fc2', 1024, (None, 1024)),
        ('stacked_lstm', 'dense', 128, (None, 128)),
        ('bidirectional_lstm', 'dense', 256, (None, 256)),
    ],
    indirect=['model'],
)
def test_attach_dense_layer(model, layer_name, output_dim, expected_output_shape):
    keras_tailor = KerasTailor(model)
    model = keras_tailor.to_embedding_model(
        freeze=True, layer_name=layer_name, output_dim=output_dim
    )
    num_layers_before = len(model.layers)
    if output_dim:
        assert len(model.layers) == num_layers_before
        assert isinstance(model.layers[-1], tf.keras.layers.Dense)
    assert model.layers[-1].trainable is True
    assert model.output_shape == expected_output_shape


@pytest.mark.parametrize(
    'model',
    [
        'dense_model',
        'simple_cnn_model',
        'vgg16_cnn_model',
        'stacked_lstm',
        'bidirectional_lstm',
    ],
    indirect=['model'],
)
@pytest.mark.parametrize('freeze', [True, False])
def test_freeze(model, freeze):
    keras_tailor = KerasTailor(model)
    for layer in model.layers:
        assert layer.trainable
    model = keras_tailor.to_embedding_model(freeze=freeze)
    for idx, layer in enumerate(model.layers):
        if freeze:
            if idx == len(model.layers) - 1:
                assert layer.trainable
            else:
                assert not layer.trainable
        else:
            assert layer.trainable


def test_keras_model_parser():
    user_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28), name='l1'),
            tf.keras.layers.Dense(128, activation='relu', name='l2'),
            tf.keras.layers.Dense(32, name='l3'),
        ]
    )

    keras_tailor = KerasTailor(user_model)

    r = keras_tailor.embedding_layers
    assert len(r) == 3
    assert r[0]['name'] == 'l1'
    assert r[1]['name'] == 'l2'
    assert r[2]['name'] == 'l3'

    assert r[0]['output_features'] == 784
    assert r[0]['nb_params'] == 0

    assert r[1]['output_features'] == 128
    assert r[1]['nb_params'] == 100480

    assert r[2]['output_features'] == 32
    assert r[2]['nb_params'] == 4128
