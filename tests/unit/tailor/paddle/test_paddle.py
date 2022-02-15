import numpy as np
import paddle
import paddle.nn as nn
import pytest

from finetuner.excepts import DeviceError
from finetuner.tailor.paddle import PaddleTailor


class LastCellPD(paddle.nn.Layer):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


@pytest.mark.parametrize(
    'paddle_model, layer_name, input_size, input_dtype',
    [
        ('paddle_dense_model', 'random_name', (128,), 'float32'),
        (
            'paddle_simple_cnn_model',
            'random_name',
            (1, 28, 28),
            'float32',
        ),
        (
            'paddle_vgg16_cnn_model',
            'random_name',
            (3, 224, 224),
            'float32',
        ),
        ('paddle_stacked_lstm', 'random_name', (128,), 'int64'),
        ('paddle_bidirectional_lstm', 'random_name', (128,), 'int64'),
    ],
    indirect=['paddle_model'],
)
def test_trim_fail_given_unexpected_layer_idx(
    paddle_model, layer_name, input_size, input_dtype
):
    with pytest.raises(KeyError):
        paddle_tailor = PaddleTailor(
            model=paddle_model,
            input_size=input_size,
            input_dtype=input_dtype,
        )
        paddle_tailor.to_embedding_model(freeze=False, layer_name=layer_name)


@pytest.mark.parametrize(
    'paddle_model, layer_name, input_size, input_dtype',
    [
        ('paddle_dense_model', 10, (128,), 'float32'),
        (
            'paddle_simple_cnn_model',
            2,
            (1, 28, 28),
            'float32',
        ),
        (
            'paddle_vgg16_cnn_model',
            4,
            (3, 224, 224),
            'float32',
        ),
        ('paddle_stacked_lstm', 10, (128,), 'int64'),
        ('paddle_bidirectional_lstm', 5, (128,), 'int64'),
    ],
    indirect=['paddle_model'],
)
@pytest.mark.parametrize('freeze', [True, False])
def test_freeze(paddle_model, layer_name, input_size, input_dtype, freeze):
    paddle_tailor = PaddleTailor(
        model=paddle_model,
        input_size=input_size,
        input_dtype=input_dtype,
    )
    model = paddle_tailor.to_embedding_model(freeze=freeze)
    if freeze:
        assert set(param.trainable for param in model.parameters()) == {False}
    else:
        assert set(param.trainable for param in model.parameters()) == {True}


@pytest.mark.parametrize(
    'paddle_model, layer_name, input_size, input_dtype, freeze_layers',
    [
        ('paddle_dense_model', 10, (128,), 'float32', ['linear_1', 'linear_5']),
        (
            'paddle_simple_cnn_model',
            2,
            (1, 28, 28),
            'float32',
            ['conv2d_1', 'maxpool2d_5'],
        ),
        (
            'paddle_vgg16_cnn_model',
            4,
            (3, 224, 224),
            'float32',
            ['conv2d_27', 'maxpool2d_31', 'adaptiveavgpool2d_32'],
        ),
        (
            'paddle_stacked_lstm',
            10,
            (128,),
            'int64',
            ['linear_layer_1', 'linear_layer_2'],
        ),
        ('paddle_bidirectional_lstm', 5, (128,), 'int64', ['lastcell_3', 'linear_4']),
    ],
    indirect=['paddle_model'],
)
def test_freeze_given_freeze_layers(
    paddle_model, layer_name, input_size, input_dtype, freeze_layers
):
    pytorch_tailor = PaddleTailor(
        model=paddle_model,
        input_size=input_size,
        input_dtype=input_dtype,
    )
    model = pytorch_tailor.to_embedding_model(freeze=freeze_layers)
    for layer, param in zip(pytorch_tailor.embedding_layers, model.parameters()):
        layer_name = layer['name']
        if layer_name in freeze_layers:
            assert not param.trainable
        else:
            assert param.trainable


def test_freeze_given_bottleneck_model_and_freeze_is_true(paddle_simple_cnn_model):
    class _BottleneckModel(nn.Layer):
        def __init__(self):
            super().__init__()
            self._linear_should_not_freeze = nn.Linear(in_features=128, out_features=64)

        def forward(self, input_):
            return self._linear(input_)

    paddle_tailor = PaddleTailor(
        model=paddle_simple_cnn_model,
        input_size=(1, 28, 28),
        input_dtype='float32',
    )

    model = paddle_tailor.to_embedding_model(
        freeze=True, layer_name='linear_8', projection_head=_BottleneckModel()
    )
    # assert bottleneck model is not freezed
    for name, param in model.named_parameters():
        if '_linear_should_not_freeze' in name:
            assert param.trainable
        else:
            assert not param.trainable


@pytest.mark.parametrize(
    'paddle_model, layer_name, input_size, input_, input_dtype, expected_output_shape',
    [
        ('paddle_dense_model', 'linear_7', (128,), (1, 128), 'float32', [1, 10]),
        (
            'paddle_simple_cnn_model',
            'dropout_9',
            (1, 28, 28),
            (1, 1, 28, 28),
            'float32',
            [1, 128],
        ),
        (
            'paddle_vgg16_cnn_model',
            'linear_36',
            (3, 224, 224),
            (1, 3, 224, 224),
            'float32',
            [1, 4096],
        ),
        ('paddle_stacked_lstm', 'linear_3', (128,), (1, 128), 'int64', [1, 256]),
        ('paddle_bidirectional_lstm', 'linear_4', (128,), (1, 128), 'int64', [1, 32]),
        ('paddle_dense_model', None, (128,), (1, 128), 'float32', [1, 10]),
        (
            'paddle_simple_cnn_model',
            None,
            (1, 28, 28),
            (1, 1, 28, 28),
            'float32',
            [1, 10],
        ),
        (
            'paddle_vgg16_cnn_model',
            None,
            (3, 224, 224),
            (1, 3, 224, 224),
            'float32',
            [1, 1000],
        ),
        ('paddle_stacked_lstm', None, (128,), (1, 128), 'int64', [1, 5]),
        ('paddle_bidirectional_lstm', None, (128,), (1, 128), 'int64', [1, 32]),
    ],
    indirect=['paddle_model'],
)
def test_to_embedding_model(
    paddle_model, layer_name, input_size, input_, input_dtype, expected_output_shape
):
    weight = paddle_model.parameters()[0].numpy()  # weight of the 0th layer
    paddle_tailor = PaddleTailor(
        model=paddle_model,
        input_size=input_size,
        input_dtype=input_dtype,
    )
    model = paddle_tailor.to_embedding_model(freeze=False, layer_name=layer_name)
    weight_after_convert = model.parameters()[0].numpy()
    np.testing.assert_array_equal(weight, weight_after_convert)
    out = model(paddle.cast(paddle.rand(input_), input_dtype))
    assert list(out.shape) == expected_output_shape


@pytest.mark.gpu
def test_to_embedding_model_with_cuda_tensor(paddle_simple_cnn_model):

    model = paddle_simple_cnn_model.to(paddle.CUDAPlace(0))

    with pytest.raises(DeviceError):
        paddle_tailor = PaddleTailor(
            model, input_size=(1, 28, 28), input_dtype='float32', device='cuda'
        )

        model = paddle_tailor.to_embedding_model()


def test_paddle_lstm_model_parser():
    user_model = paddle.nn.Sequential(
        paddle.nn.Embedding(num_embeddings=5000, embedding_dim=64),
        paddle.nn.LSTM(64, 64, direction='bidirectional'),
        LastCellPD(),
        paddle.nn.Linear(in_features=2 * 64, out_features=32),
    )
    paddle_tailor = PaddleTailor(
        model=user_model,
        input_size=(5000,),
        input_dtype='int64',
    )
    r = paddle_tailor.embedding_layers
    assert len(r) == 3

    # flat layer can be a nonparametric candidate
    assert r[0]['output_features'] == 64
    assert r[0]['nb_params'] == 320000

    assert r[1]['output_features'] == 128
    assert r[1]['nb_params'] == 0

    assert r[2]['output_features'] == 32
    assert r[2]['nb_params'] == 4128


def test_paddle_mlp_model_parser():
    user_model = paddle.nn.Sequential(
        paddle.nn.Flatten(),
        paddle.nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        paddle.nn.ReLU(),
        paddle.nn.Linear(in_features=128, out_features=32),
    )
    paddle_tailor = PaddleTailor(
        model=user_model,
        input_size=(28, 28),
        input_dtype='float32',
    )
    r = paddle_tailor.embedding_layers
    assert len(r) == 4

    # flat layer can be a nonparametric candidate
    assert r[0]['output_features'] == 784
    assert r[0]['nb_params'] == 0

    assert r[1]['output_features'] == 128
    assert r[1]['nb_params'] == 100480

    # relu layer is a nonparametric candidate
    assert r[2]['output_features'] == 128
    assert r[2]['nb_params'] == 0

    assert r[3]['output_features'] == 32
    assert r[3]['nb_params'] == 4128
