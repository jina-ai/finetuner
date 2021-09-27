import pytest
import tensorflow as tf

from finetuner.tailor.keras import trim, freeze


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
def lstm_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(1000, 128, input_length=64))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


@pytest.fixture(
    params=['dense_model', 'simple_cnn_model', 'vgg16_cnn_model', 'lstm_model']
)
def model(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    'model, layer_idx',
    [
        ('dense_model', 10),  # 10th layer does not exist
        ('simple_cnn_model', 2),  # 2nd layer is a convolutional layer
        ('vgg16_cnn_model', 4),  # 4th layer is a convolutional layer
        ('lstm_model', 10),  # 10th layer does not exist
    ],
    indirect=['model'],
)
def test_trim_fail_given_unexpected_layer_idx(model, layer_idx):
    with pytest.raises(IndexError):
        trim(model, layer_idx=layer_idx)


@pytest.mark.parametrize(
    'model, layer_idx, expected_output_shape',
    [
        ('dense_model', 3, (None, 32)),
        ('simple_cnn_model', 5, (None, 9216)),
        ('vgg16_cnn_model', 21, (None, 4096)),
        ('lstm_model', 2, (None, 64)),
    ],
    indirect=['model'],
)
def test_trim(model, layer_idx, expected_output_shape):
    model = trim(model=model, layer_idx=layer_idx)
    assert model.output_shape == expected_output_shape


@pytest.mark.parametrize(
    'model',
    ['dense_model', 'simple_cnn_model', 'vgg16_cnn_model', 'lstm_model'],
    indirect=['model'],
)
def test_freeze(model):
    for layer in model.layers:
        assert layer.trainable
    model = freeze(model)
    for layer in model.layers:
        assert not layer.trainable
