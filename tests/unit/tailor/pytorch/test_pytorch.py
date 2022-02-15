import numpy as np
import pytest
import torch
import torch.nn as nn

from finetuner.tailor.pytorch import PytorchTailor


class LastCellPT(torch.nn.Module):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


@pytest.mark.parametrize(
    'torch_model, layer_name, input_size, input_dtype',
    [
        ('torch_dense_model', 'random_name', (128,), 'float32'),
        (
            'torch_simple_cnn_model',
            'random_name',
            (1, 28, 28),
            'float32',
        ),
        (
            'torch_vgg16_cnn_model',
            'random_name',
            (3, 224, 224),
            'float32',
        ),
        ('torch_stacked_lstm', 'random_name', (128,), 'int64'),
        ('torch_bidirectional_lstm', 'random_name', (128,), 'int64'),
    ],
    indirect=['torch_model'],
)
def test_trim_fail_given_unexpected_layer_idx(
    torch_model, layer_name, input_size, input_dtype
):
    with pytest.raises(KeyError):
        paddle_tailor = PytorchTailor(
            model=torch_model,
            input_size=input_size,
            input_dtype=input_dtype,
            device='cuda',
        )
        paddle_tailor.to_embedding_model(
            freeze=False,
            layer_name=layer_name,
        )


@pytest.mark.parametrize(
    'torch_model, layer_name, input_size, input_, input_dtype, expected_output_shape',
    [
        ('torch_dense_model', 'linear_7', (128,), (1, 128), 'float32', [1, 10]),
        (
            'torch_simple_cnn_model',
            'dropout_9',
            (1, 28, 28),
            (1, 1, 28, 28),
            'float32',
            [1, 128],
        ),
        (
            'torch_vgg16_cnn_model',
            'linear_36',
            (3, 224, 224),
            (1, 3, 224, 224),
            'float32',
            [1, 4096],
        ),
        ('torch_stacked_lstm', 'linear_3', (128,), (1, 128), 'int64', [1, 256]),
        ('torch_bidirectional_lstm', 'linear_4', (128,), (1, 128), 'int64', [1, 32]),
        ('torch_dense_model', None, (128,), (1, 128), 'float32', [1, 10]),
        (
            'torch_simple_cnn_model',
            None,
            (1, 28, 28),
            (1, 1, 28, 28),
            'float32',
            [1, 10],
        ),
        (
            'torch_vgg16_cnn_model',
            None,
            (3, 224, 224),
            (1, 3, 224, 224),
            'float32',
            [1, 1000],
        ),
        ('torch_stacked_lstm', None, (128,), (1, 128), 'int64', [1, 5]),
        ('torch_bidirectional_lstm', None, (128,), (1, 128), 'int64', [1, 32]),
    ],
    indirect=['torch_model'],
)
def test_to_embedding_model(
    torch_model, layer_name, input_size, input_, input_dtype, expected_output_shape
):
    weights = (
        list(torch_model.parameters())[0].detach().numpy()
    )  # weights of the first layer
    pytorch_tailor = PytorchTailor(
        model=torch_model,
        input_size=input_size,
        input_dtype=input_dtype,
    )
    model = pytorch_tailor.to_embedding_model(
        freeze=False,
        layer_name=layer_name,
    )
    weights_after_convert = list(model.parameters())[0].detach().numpy()
    np.testing.assert_array_equal(weights, weights_after_convert)
    input_ = torch.rand(input_)
    if input_dtype == 'int64':
        input_ = input_.type(torch.LongTensor)
    out = model(input_)
    assert list(out.size()) == expected_output_shape


@pytest.mark.gpu
def test_to_embedding_model_with_cuda_tensor(torch_simple_cnn_model):
    device = torch.device('cuda:0')

    model = torch_simple_cnn_model.to(device)

    pytorch_tailor = PytorchTailor(
        model, input_size=(1, 28, 28), input_dtype='float32', device='cuda'
    )

    model = pytorch_tailor.to_embedding_model()
    assert model


@pytest.mark.parametrize(
    'torch_model, layer_name, input_size, input_dtype',
    [
        ('torch_dense_model', 10, (128,), 'float32'),
        (
            'torch_simple_cnn_model',
            2,
            (1, 28, 28),
            'float32',
        ),
        (
            'torch_vgg16_cnn_model',
            4,
            (3, 224, 224),
            'float32',
        ),
        ('torch_stacked_lstm', 10, (128,), 'int64'),
        ('torch_bidirectional_lstm', 5, (128,), 'int64'),
    ],
    indirect=['torch_model'],
)
@pytest.mark.parametrize('freeze', [True, False])
def test_freeze(torch_model, layer_name, input_size, input_dtype, freeze):
    pytorch_tailor = PytorchTailor(
        model=torch_model,
        input_size=input_size,
        input_dtype=input_dtype,
    )
    model = pytorch_tailor.to_embedding_model(freeze=freeze)
    if freeze:  # all freezed
        assert set(param.requires_grad for param in model.parameters()) == {False}
    else:
        assert set(param.requires_grad for param in model.parameters()) == {True}


def test_freeze_given_bottleneck_model_and_freeze_is_true(torch_simple_cnn_model):
    class _BottleneckModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._linear_should_not_freeze = nn.Linear(in_features=128, out_features=64)

        def forward(self, input_):
            return self._linear(input_)

    pytorch_tailor = PytorchTailor(
        model=torch_simple_cnn_model,
        input_size=(1, 28, 28),
        input_dtype='float32',
    )

    model = pytorch_tailor.to_embedding_model(
        freeze=True, layer_name='linear_8', projection_head=_BottleneckModel()
    )
    # assert bottleneck model is not freezed
    for name, param in model.named_parameters():
        if '_linear_should_not_freeze' in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad


@pytest.mark.parametrize(
    'torch_model, layer_name, input_size, input_dtype, freeze_layers',
    [
        ('torch_dense_model', 10, (128,), 'float32', ['linear_1', 'linear_5']),
        (
            'torch_simple_cnn_model',
            2,
            (1, 28, 28),
            'float32',
            ['conv2d_1', 'maxpool2d_5'],
        ),
        (
            'torch_vgg16_cnn_model',
            4,
            (3, 224, 224),
            'float32',
            ['conv2d_27', 'maxpool2d_31', 'adaptiveavgpool2d_32'],
        ),
        (
            'torch_stacked_lstm',
            10,
            (128,),
            'int64',
            ['linear_layer_1', 'linear_layer_2'],
        ),
        ('torch_bidirectional_lstm', 5, (128,), 'int64', ['lastcell_3', 'linear_4']),
    ],
    indirect=['torch_model'],
)
def test_freeze_given_freeze_layers(
    torch_model, layer_name, input_size, input_dtype, freeze_layers
):
    pytorch_tailor = PytorchTailor(
        model=torch_model,
        input_size=input_size,
        input_dtype=input_dtype,
    )
    model = pytorch_tailor.to_embedding_model(freeze=freeze_layers)
    for layer, param in zip(pytorch_tailor.embedding_layers, model.parameters()):
        layer_name = layer['name']
        if layer_name in freeze_layers:
            assert not param.requires_grad
        else:
            assert param.requires_grad


def test_torch_lstm_model_parser():
    user_model = torch.nn.Sequential(
        torch.nn.Embedding(num_embeddings=5000, embedding_dim=64),
        torch.nn.LSTM(64, 64, bidirectional=True, batch_first=True),
        LastCellPT(),
        torch.nn.Linear(in_features=2 * 64, out_features=32),
    )
    pytorch_tailor = PytorchTailor(
        model=user_model,
        input_size=(5000,),
        input_dtype='int64',
    )
    r = pytorch_tailor.embedding_layers
    assert len(r) == 3

    # flat layer can be a nonparametric candidate
    assert r[0]['output_features'] == 64
    assert r[0]['nb_params'] == 320000

    assert r[1]['output_features'] == 128
    assert r[1]['nb_params'] == 0


def test_torch_mlp_model_parser():
    user_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=32),
    )
    pytorch_tailor = PytorchTailor(
        model=user_model,
        input_size=(28, 28),
        input_dtype='float32',
    )
    r = pytorch_tailor.embedding_layers
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
