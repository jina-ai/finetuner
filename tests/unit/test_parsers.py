import paddle
import pytest
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertConfig

from trainer.paddle.parser import get_candidate_layers as gcl_p
from trainer.pytorch.parser import get_candidate_layers as gcl_t


@pytest.fixture
def torch_model():
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=32),
    )


@pytest.fixture
def cnn_model():
    return models.vgg16(pretrained=False)


@pytest.fixture
def transformer_model():
    return BertModel(BertConfig())


@pytest.fixture
def lstm_model():
    class Encoder(nn.Module):
        def __init__(self, seq_len=128, no_features=128, embedding_size=1024):
            super().__init__()
            self.seq_len = seq_len
            self.no_features = no_features
            self.embedding_size = embedding_size
            self.hidden_size = 2 * embedding_size
            self.lstm = nn.LSTM(
                input_size=self.no_features,
                hidden_size=embedding_size,
                num_layers=3,
                batch_first=True,
            )

        def forward(self, x):
            x, (hidden_state, cell_state) = self.lstm(x)
            last_lstm_layer_hidden_state = hidden_state[-1, :, :]
            return last_lstm_layer_hidden_state

    return Encoder()


@pytest.mark.parametrize(
    'user_model, parser',
    [
        (
            paddle.nn.Sequential(
                paddle.nn.Flatten(),
                paddle.nn.Linear(
                    in_features=28 * 28,
                    out_features=128,
                ),
                paddle.nn.ReLU(),
                paddle.nn.Linear(in_features=128, out_features=32),
            ),
            gcl_p,
        ),
        (
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(
                    in_features=28 * 28,
                    out_features=128,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=128, out_features=32),
            ),
            gcl_t,
        ),
    ],
)
def test_paddle_torch_model_parser(user_model, parser):
    r = parser(user_model, input_size=(1, 28, 28))
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


def test_keras_model_parser():
    user_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28), name='l1'),
            tf.keras.layers.Dense(128, activation='relu', name='l2'),
            tf.keras.layers.Dense(32, name='l3'),
        ]
    )

    from trainer.keras.parser import get_candidate_layers

    r = get_candidate_layers(user_model)
    assert len(r) == 3
    assert r[0]['name'] == 'l1'
    assert r[1]['name'] == 'l2'
    assert r[2]['name'] == 'l3'

    # flat layer can be a nonparametric candidate
    assert r[0]['output_features'] == 784
    assert r[0]['params'] == 0

    assert r[1]['output_features'] == 128
    assert r[1]['params'] == 100480

    assert r[2]['output_features'] == 32
    assert r[2]['params'] == 4128


def test_parse_torch_given_vision_model(cnn_model):
    input_size = (3, 224, 224)
    r = gcl_t(cnn_model, input_size)
    assert len(r) == 8  # all layers inside classifier
    for item in r[:-2]:
        assert item['output_features'] == 4096
        if item['cls_name'] == 'Linear':
            assert item['params'] > 0
        else:
            assert item['params'] == 0
    for item in r[-2:]:  # Last 2 layers classification layer map to 1000 dim.
        assert item['output_features'] == 1000
        if item['cls_name'] == 'Linear':
            assert item['params'] > 0
        else:
            assert item['params'] == 0


def test_parse_torch_given_lstm_model(lstm_model):
    input_size = 128  # seq
    r = gcl_t(lstm_model, input_size=(1, input_size))
    assert len(r) == 2
    assert r[1]['output_features'] == 1024
    assert r[1]['params'] == 0


def test_parse_torch_given_transformer_model(transformer_model):
    input_size = 128
    r = gcl_t(transformer_model, input_size=(input_size,), dtype=torch.IntTensor)
    assert len(r) == 3
    for item in r:
        assert item['output_features'] == 768
        if item['cls_name'] == 'Linear':
            assert item['params'] > 0
        else:
            assert item['params'] == 0
