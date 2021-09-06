import pytest
import tensorflow as tf

from finetuner.tailor.keras import trim


@pytest.fixture
def dense_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(16,)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


@pytest.fixture
def simple_cnn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
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


@pytest.mark.parametrize('freeze', [True, False])
@pytest.mark.parametrize(
    'model, layer_idx, expected_output_shape',
    [
        ('dense_model', 1, (None, 64)),
        ('simple_cnn_model', 4, (None, 1600)),
        ('vgg16_cnn_model', 20, (None, 4096)),
        ('lstm_model', 1, (None, 64)),
    ],
    indirect=['model'],
)
def test_trim(model, layer_idx, expected_output_shape, freeze):
    model = trim(model=model, layer_idx=layer_idx, freeze=freeze)
    assert model.output_shape == expected_output_shape
    if freeze:
        for layer in model.layers:
            assert layer.trainable is False
