import pytest
import torch
import torch.nn as nn

from finetuner.tailor.pytorch import trim


@pytest.fixture
def dense_model():
    return torch.nn.Sequential(
        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=10),
        nn.Softmax(),
    )


@pytest.fixture
def simple_cnn_model():
    return torch.nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(in_features=196, out_features=10),
        nn.Softmax(),
    )


@pytest.fixture
def vgg16_cnn_model():
    import torchvision.models as models

    return models.vgg16(pretrained=False)


#
#
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
                num_layers=1,
                batch_first=True,
            )

        def forward(self, x):
            x, (hidden_state, cell_state) = self.lstm(x)
            last_lstm_layer_hidden_state = hidden_state[-1, :, :]
            return last_lstm_layer_hidden_state

    return Encoder()


@pytest.fixture(params=['dense_model'])
def model(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    'model, layer_idx, input_size',
    [
        ('dense_model', 10, (128,)),  # 10th layer does not exist
        ('simple_cnn_model', 2, (1, 28, 28)),  # 2nd layer is a convolutional layer
        ('vgg16_cnn_model', 4, (3, 224, 224)),  # 4th layer is a convolutional layer
        ('lstm_model', 10, (1, 128)),  # 10th layer does not exist
    ],
    indirect=['model'],
)
def test_trim_fail_given_unexpected_layer_idx(model, layer_idx, input_size):
    with pytest.raises(IndexError):
        trim(model, layer_idx=layer_idx, input_size=input_size)


@pytest.mark.parametrize('freeze', [True, False])
@pytest.mark.parametrize(
    'model, layer_idx, input_size, expected_output_shape',
    [
        ('dense_model', 3, (128,), 32),
        ('simple_cnn_model', 10, (1, 28, 28), (None, 1600)),
        ('vgg16_cnn_model', 33, (3, 224, 224), (None, 4096)),
        ('lstm_model', 1, (1, 128), 128),
    ],
    indirect=['model'],
)
def test_trim(model, layer_idx, input_size, expected_output_shape, freeze):
    model = trim(model=model, layer_idx=layer_idx, freeze=freeze, input_size=input_size)
    print(model)
    if freeze:
        for param in model.parameters():
            assert param.requires_grad is False
