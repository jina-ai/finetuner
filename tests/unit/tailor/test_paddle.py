import pytest

import paddle

from finetuner.tailor.paddle import trim, freeze


@pytest.fixture
def vgg16_cnn_model():
    return paddle.vision.models.vgg16(pretrained=False)


@pytest.fixture(params=['vgg16_cnn_model'])
def model(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    'model',
    ['vgg16_cnn_model'],
    indirect=['model'],
)
def test_freeze(model):
    for param in model.parameters():
        assert param.trainable
    model = freeze(model)
    for param in model.parameters():
        assert not param.trainable
