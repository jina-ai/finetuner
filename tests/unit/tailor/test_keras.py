import pytest
import tensorflow as tf

from finetuner.tailor.keras import tail


@pytest.fixture
def dense_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(16,)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
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
    model.add(tf.keras.layers.Embedding(1000, 128, 300))
    model.add(tf.keras.layers.LSTM(64, return_sequence=True))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


@pytest.fixture(
    params=['dense_model', 'simple_cnn_model', 'vgg16_cnn_model', 'lstm_model']
)
def model(request):
    return request.getfixturevalue(request.param)


def test_tail_fail_with_unexpected_layer_idx(dense_model):
    with pytest.raises(IndexError):
        tail(dense_model, layer_idx=10)


@pytest.mark.parametrize(
    'model, expected',
    [
        ('dense_model', 1),
        # ('simple_cnn_model', 1),
        # ('vgg16_cnn_model', 1),
        # ('lstm_model', 1),
    ],
    indirect=['model'],
)
def test_tail(model, expected):
    print(model)
    print(type(model))
    tail(model=model, layer_idx=1, freeze=True)
