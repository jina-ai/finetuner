import pytest

import paddle
import paddle.nn as nn

from finetuner.tailor.paddle import trim, freeze


@pytest.fixture
def dense_model():
    return nn.Sequential(
        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=10),
        nn.Softmax(),
    )


@pytest.fixture
def simple_cnn_model():
    return nn.Sequential(
        nn.Conv2D(1, 4, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2D(4),
        nn.ReLU(),
        nn.MaxPool2D(kernel_size=2, stride=2),
        nn.Conv2D(4, 4, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2D(4),
        nn.ReLU(),
        nn.MaxPool2D(kernel_size=2, stride=2),
        nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(in_features=196, out_features=10),
        nn.Softmax(),
    )


@pytest.fixture
def vgg16_cnn_model():
    return paddle.vision.models.vgg16(pretrained=False)


@pytest.fixture(params=['dense_model', 'simple_cnn_model', 'vgg16_cnn_model'])
def model(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    'model',
    ['dense_model', 'simple_cnn_model', 'vgg16_cnn_model'],
    indirect=['model'],
)
def test_freeze(model):
    for param in model.parameters():
        if not param.stop_gradient:
            assert param.trainable
    model = freeze(model)
    for param in model.parameters():
        assert not param.trainable


def test_trim(vgg16_cnn_model):
    model = trim(vgg16_cnn_model, layer_idx=33, input_size=(3, 224, 224))
    print(model)
