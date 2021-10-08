import paddle
import paddle.nn as nn
import pytest
import numpy as np

from finetuner.tailor.paddle import PaddleTailor


class LastCellPD(paddle.nn.Layer):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


@pytest.fixture
def dense_model():
    return nn.Sequential(
        nn.Linear(in_features=128, out_features=128),
        nn.ReLU(),
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
        nn.Conv2D(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2D(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2D(2),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(9216, 128),
        nn.Dropout(0.25),
        nn.Linear(128, 10),
        nn.Softmax(),
    )


@pytest.fixture
def vgg16_cnn_model():
    return paddle.vision.models.vgg16(pretrained=False)


@pytest.fixture
def stacked_lstm():
    class LSTMClassifier(nn.Layer):
        """A simple LSTM for text classification."""

        def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
            self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, num_layers=3)
            self.linear_layer_1 = nn.Linear(hidden_dim, hidden_dim)
            self.relu_layer = nn.ReLU()
            self.linear_layer_2 = nn.Linear(hidden_dim, target_size)
            self.classification_layer = nn.Softmax(1)

        def forward(self, input_):
            embedding = self.embedding_layer(input_)
            lstm_out, _ = self.lstm_layer(embedding)
            # lstm_out -> (batch_size * seq_len * hidden_dim)
            last_lstm_out = lstm_out[:, -1, :]
            # last_lstm_out -> (1, hidden_dim)
            linear_out_1 = self.linear_layer_1(last_lstm_out)
            relu_out = self.relu_layer(linear_out_1)
            linear_out_2 = self.linear_layer_2(relu_out)
            classification_out = self.classification_layer(linear_out_2)
            return classification_out

    return LSTMClassifier(1024, 256, 1000, 5)


@pytest.fixture
def bidirectional_lstm():
    class LastCell(nn.Layer):
        def forward(self, x):
            out, _ = x
            return out[:, -1, :]

    return nn.Sequential(
        nn.Embedding(num_embeddings=5000, embedding_dim=64),
        nn.LSTM(64, 64, direction='bidirectional'),
        LastCell(),
        nn.Linear(in_features=128, out_features=32),
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
    'model, layer_name, input_size, input_dtype',
    [
        ('dense_model', 'random_name', (128,), 'float32'),
        (
            'simple_cnn_model',
            'random_name',
            (1, 28, 28),
            'float32',
        ),
        (
            'vgg16_cnn_model',
            'random_name',
            (3, 224, 224),
            'float32',
        ),
        ('stacked_lstm', 'random_name', (128,), 'int64'),
        ('bidirectional_lstm', 'random_name', (128,), 'int64'),
    ],
    indirect=['model'],
)
def test_trim_fail_given_unexpected_layer_idx(
    model, layer_name, input_size, input_dtype
):
    with pytest.raises(KeyError):
        paddle_tailor = PaddleTailor(
            model=model,
            input_size=input_size,
            input_dtype=input_dtype,
        )
        paddle_tailor.to_embedding_model(freeze=False, layer_name=layer_name)


@pytest.mark.parametrize(
    'model, layer_name, input_size, input_dtype',
    [
        ('dense_model', 10, (128,), 'float32'),
        (
            'simple_cnn_model',
            2,
            (1, 28, 28),
            'float32',
        ),
        (
            'vgg16_cnn_model',
            4,
            (3, 224, 224),
            'float32',
        ),
        ('stacked_lstm', 10, (128,), 'int64'),
        ('bidirectional_lstm', 5, (128,), 'int64'),
    ],
    indirect=['model'],
)
@pytest.mark.parametrize('freeze', [True, False])
def test_freeze(model, layer_name, input_size, input_dtype, freeze):
    paddle_tailor = PaddleTailor(
        model=model,
        input_size=input_size,
        input_dtype=input_dtype,
    )
    model = paddle_tailor.to_embedding_model(freeze=freeze, output_dim=2)
    if freeze:
        assert len(set(param.trainable for param in model.parameters())) == 2
    else:
        assert set(param.trainable for param in model.parameters()) == {True}


@pytest.mark.parametrize(
    'model, layer_name, input_size, input_, input_dtype, expected_output_shape',
    [
        ('dense_model', 'linear_7', (128,), (1, 128), 'float32', [1, 10]),
        (
            'simple_cnn_model',
            'dropout_9',
            (1, 28, 28),
            (1, 1, 28, 28),
            'float32',
            [1, 128],
        ),
        (
            'vgg16_cnn_model',
            'linear_36',
            (3, 224, 224),
            (1, 3, 224, 224),
            'float32',
            [1, 4096],
        ),
        ('stacked_lstm', 'linear_3', (128,), (1, 128), 'int64', [1, 256]),
        ('bidirectional_lstm', 'linear_4', (128,), (1, 128), 'int64', [1, 32]),
        ('dense_model', None, (128,), (1, 128), 'float32', [1, 10]),
        (
            'simple_cnn_model',
            None,
            (1, 28, 28),
            (1, 1, 28, 28),
            'float32',
            [1, 10],
        ),
        (
            'vgg16_cnn_model',
            None,
            (3, 224, 224),
            (1, 3, 224, 224),
            'float32',
            [1, 1000],
        ),
        ('stacked_lstm', None, (128,), (1, 128), 'int64', [1, 5]),
        ('bidirectional_lstm', None, (128,), (1, 128), 'int64', [1, 32]),
    ],
    indirect=['model'],
)
def test_to_embedding_model(
    model, layer_name, input_size, input_, input_dtype, expected_output_shape
):
    weight = model.parameters()[0].numpy()  # weight of the 0th layer
    paddle_tailor = PaddleTailor(
        model=model,
        input_size=input_size,
        input_dtype=input_dtype,
    )
    model = paddle_tailor.to_embedding_model(freeze=False, layer_name=layer_name)
    weight_after_convert = model.parameters()[0].numpy()
    np.testing.assert_array_equal(weight, weight_after_convert)
    out = model(paddle.cast(paddle.rand(input_), input_dtype))
    assert list(out.shape) == expected_output_shape


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
    assert len(r) == 2

    # flat layer can be a nonparametric candidate
    assert r[0]['output_features'] == 128
    assert r[0]['nb_params'] == 0

    assert r[1]['output_features'] == 32
    assert r[1]['nb_params'] == 4128


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
