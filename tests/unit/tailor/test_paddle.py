import pytest

import paddle
import paddle.nn as nn

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
            freeze=False,
            embedding_layer_name=layer_name,
            input_size=input_size,
            input_dtype=input_dtype,
        )
        paddle_tailor._trim()


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
def test_freeze(model, layer_name, input_size, input_dtype):
    paddle_tailor = PaddleTailor(
        model=model,
        freeze=False,
        embedding_layer_name=layer_name,
        input_size=input_size,
        input_dtype=input_dtype,
    )
    for param in paddle_tailor.model.parameters():
        if not param.stop_gradient:
            assert param.trainable
    paddle_tailor._freeze_weights()
    for param in paddle_tailor.model.parameters():
        assert not param.trainable


@pytest.mark.parametrize(
    'model, layer_name, input_size, input_, input_dtype, expected_output_shape',
    [
        ('dense_model', 'linear_27', (128,), (1, 128), 'float32', [1, 32]),
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
            'linear_31',
            (3, 224, 224),
            (1, 3, 224, 224),
            'float32',
            [1, 4096],
        ),
        ('stacked_lstm', 'linear_33', (128,), (1, 128), 'int64', [1, 256]),
        ('bidirectional_lstm', 'linear_35', (128,), (1, 128), 'int64', [1, 128]),
    ],
    indirect=['model'],
)
def test_trim(
    model, layer_name, input_size, input_, input_dtype, expected_output_shape
):
    paddle_tailor = PaddleTailor(
        model=model,
        freeze=False,
        embedding_layer_name=layer_name,
        input_size=input_size,
        input_dtype=input_dtype,
    )
    paddle_tailor._trim()
    out = paddle_tailor.model(paddle.cast(paddle.rand(input_), input_dtype))
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
        freeze=False,
        embedding_layer_name='last_cell_pd_0',
        input_size=(5000,),
        input_dtype='int64',
    )
    r = paddle_tailor.embedding_layers
    assert len(r) == 2

    # flat layer can be a nonparametric candidate
    assert r[0]['output_features'] == 128
    assert r[0]['params'] == 0

    assert r[1]['output_features'] == 32
    assert r[1]['params'] == 4128


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
        freeze=False,
        embedding_layer_name='linear_1',
        input_size=(28, 28),
        input_dtype='float32',
    )
    r = paddle_tailor.embedding_layers
    assert len(r) == 4

    # flat layer can be a nonparametric candidate
    assert r[0]['output_features'] == 784
    assert r[0]['params'] == 0

    assert r[1]['output_features'] == 128
    assert r[1]['params'] == 100480

    # relu layer is a nonparametric candidate
    assert r[2]['output_features'] == 128
    assert r[2]['params'] == 0

    assert r[3]['output_features'] == 32
    assert r[3]['params'] == 4128
