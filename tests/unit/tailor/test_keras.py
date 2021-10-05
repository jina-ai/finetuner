import pytest
import tensorflow as tf

from finetuner.tailor.keras import KerasTailor


@pytest.fixture(autouse=True)
def clear_session():
    tf.keras.backend.clear_session()


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
    return tf.keras.applications.vgg16.VGG16(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
    )


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
        keras_tailor = KerasTailor(model, True, layer_name)
        keras_tailor._trim()


@pytest.mark.parametrize(
    'model, layer_name, expected_output_shape',
    [
        ('dense_model', 'dense_2', (None, 32)),
        ('simple_cnn_model', 'flatten', (None, 9216)),
        ('vgg16_cnn_model', 'fc1', (None, 4096)),
        ('stacked_lstm', 'lstm_2', (None, 256)),
        ('bidirectional_lstm', 'bidirectional', (None, 128)),
    ],
    indirect=['model'],
)
def test_trim(model, layer_name, expected_output_shape):
    keras_tailor = KerasTailor(model, False, layer_name)
    keras_tailor._trim()
    assert keras_tailor.model.output_shape == expected_output_shape


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
def test_freeze(model):
    keras_tailor = KerasTailor(model)
    for layer in keras_tailor.model.layers:
        assert layer.trainable
    keras_tailor._freeze_weights()
    for layer in keras_tailor.model.layers:
        assert not layer.trainable


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

    # flat layer can be a nonparametric candidate
    assert r[0]['output_features'] == 784
    assert r[0]['params'] == 0

    assert r[1]['output_features'] == 128
    assert r[1]['params'] == 100480

    assert r[2]['output_features'] == 32
    assert r[2]['params'] == 4128
