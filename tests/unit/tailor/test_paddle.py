import pytest
import paddle

from finetuner.tailor.paddle import trim


@pytest.fixture
def dense_model():
    return paddle.nn.Sequential(
        paddle.nn.Linear(64, 128),
        paddle.nn.ReLU(),
        paddle.nn.Linear(128, 32),
        paddle.nn.ReLU(),
        paddle.nn.Linear(32, 10),
        paddle.nn.ReLU(),
        paddle.nn.Softmax(),
    )


@pytest.fixture(params=['dense_model'])
def model(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize('freeze', [True, False])
@pytest.mark.parametrize(
    'model, layer_idx, expected_output_shape',
    [
        ('dense_model', 1, (None, 64)),
    ],
    indirect=['model'],
)
def test_trim(model, layer_idx, expected_output_shape, freeze):
    model = trim(model=model, input_size=(64,), layer_idx=layer_idx, freeze=freeze)
    assert model.output_features == expected_output_shape
    if freeze:
        for layer in model.layers:
            assert layer.trainable is False
