import pytest
import torch.nn as nn
import torchvision.models as models

from trainer.pytorch.parser import parse, get_candidate_layers


@pytest.fixture
def cnn_model():
    return models.vgg16(pretrained=False)


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


def test_parse_vision_model(cnn_model):
    layer_index = 35
    input_size = (3, 224, 224)
    out_features = 4096
    candidate_layers = get_candidate_layers(cnn_model, input_size)
    assert len(candidate_layers) == 8  # all layers inside classifier
    parsed_model = parse(cnn_model, input_size=input_size, layer_index=layer_index)
    assert len(parsed_model) == layer_index + 1
    assert parsed_model.top  # assure we have a last layer named top
    assert parsed_model.top.out_features == out_features


def test_parse_lstm_model(lstm_model):
    input_size = 128  # seq
    num_layers = 3  # stacked 3 layer LSTM
    candidate_layers = get_candidate_layers(lstm_model, input_size=(1, input_size))
    assert len(candidate_layers) == 1
    parsed_model = parse(lstm_model, input_size=(1, input_size), layer_index=1)
    for child in parsed_model.children():
        assert child.input_size == input_size
        assert child.hidden_size == 1024
        assert child.num_layers == num_layers
