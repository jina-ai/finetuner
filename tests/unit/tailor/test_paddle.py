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


@pytest.fixture
def lstm_model():
    class Encoder(nn.Layer):
        def __init__(self, seq_len=128, no_features=128, embedding_size=1024):
            super().__init__()
            self.seq_len = seq_len
            self.no_features = no_features
            self.embedding_size = embedding_size
            self.hidden_size = 2 * embedding_size
            self.lstm = nn.LSTM(
                input_size=self.no_features,
                hidden_size=embedding_size,
                num_layers=1,
            )
            self.fc = nn.Linear(1024, 10)

        def forward(self, x):
            x, (hidden_state, cell_state) = self.lstm(x)
            last_lstm_layer_hidden_state = hidden_state[-1, :, :]
            pred = self.fc(last_lstm_layer_hidden_state)
            return pred

    return Encoder()


@pytest.fixture(
    params=['dense_model', 'simple_cnn_model', 'vgg16_cnn_model', 'lstm_model']
)
def model(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    'model',
    ['dense_model', 'simple_cnn_model', 'vgg16_cnn_model', 'lstm_model'],
    indirect=['model'],
)
def test_freeze(model):
    for param in model.parameters():
        if not param.stop_gradient:
            assert param.trainable
    model = freeze(model)
    for param in model.parameters():
        assert not param.trainable


@pytest.mark.parametrize(
    'model, layer_idx, input_size, input_, expected_output_shape',
    [
        ('dense_model', 4, (128,), (1, 128), [1, 32]),
        ('simple_cnn_model', 11, (1, 28, 28), (1, 1, 28, 28), [1, 10]),
        ('vgg16_cnn_model', 35, (3, 224, 224), (1, 3, 224, 224), [1, 4096]),
        ('lstm_model', 1, (1, 128), (1, 1, 128), [1, 1024]),
    ],
    indirect=['model'],
)
def test_trim(model, layer_idx, input_size, input_, expected_output_shape):
    model = trim(model=model, layer_idx=layer_idx, input_size=input_size)
    out = model(paddle.rand(input_))
    assert list(out.shape) == expected_output_shape
