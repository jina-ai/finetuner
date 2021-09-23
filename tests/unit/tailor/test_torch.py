import pytest
import torch
import torch.nn as nn

from finetuner.tailor.pytorch import trim


@pytest.fixture
def dense_model():
    return torch.nn.Sequential(
        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=10),
        nn.Softmax(),
    )


# @pytest.fixture
# def simple_cnn_model():
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.Input(shape=(28, 28, 1)))
#     model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dropout(0.2))
#     model.add(tf.keras.layers.Dense(10, activation="softmax"))
#     return model
#
#
@pytest.fixture
def vgg16_cnn_model():
    import torchvision.models as models

    return models.vgg16(pretrained=False)


#
#
# @pytest.fixture
# def lstm_model():
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Embedding(1000, 128, input_length=64))
#     model.add(tf.keras.layers.LSTM(64))
#     model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#     return model


@pytest.fixture(params=['dense_model'])
def model(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    'model, layer_idx',
    [
        ('dense_model', 10),  # 10th layer does not exist
        # ('simple_cnn_model', 2),  # 2nd layer is a convolutional layer
        # ('vgg16_cnn_model', 4),  # 4th layer is a convolutional layer
        # ('lstm_model', 10),  # 10th layer does not exist
    ],
    indirect=['model'],
)
def test_trim_fail_given_unexpected_layer_idx(model, layer_idx):
    with pytest.raises(IndexError):
        trim(model, layer_idx=layer_idx)


@pytest.mark.parametrize('freeze', [True, False])
@pytest.mark.parametrize(
    'model, layer_idx, input_size, expected_output_shape',
    [
        ('dense_model', 2, (128,), 32),
        # ('simple_cnn_model', 4, (None, 1600)),
        # ('vgg16_cnn_model', 32, (3, 224, 224), (None, 4096)),
        # ('lstm_model', 1, (None, 64)),
    ],
    indirect=['model'],
)
def test_trim(model, layer_idx, input_size, expected_output_shape, freeze):
    model = trim(model=model, layer_idx=layer_idx, freeze=freeze, input_size=input_size)
    assert model[layer_idx].out_features == expected_output_shape
    if freeze:
        for param in model.parameters():
            assert param.requires_grad == False
