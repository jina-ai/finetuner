import pytest
import torch


class LastCellPT(torch.nn.Module):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


@pytest.fixture
def torch_dense_model():
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=128, out_features=128),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=64, out_features=32),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=32, out_features=10),
        torch.nn.Softmax(),
    )


@pytest.fixture
def torch_simple_cnn_model():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3, 1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3, 1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Dropout(0.25),
        torch.nn.Flatten(),
        torch.nn.Linear(9216, 128),
        torch.nn.Dropout(0.25),
        torch.nn.Linear(128, 10),
        torch.nn.Softmax(),
    )


@pytest.fixture
def torch_vgg16_cnn_model():
    import torchvision.models as models

    return models.vgg16(pretrained=False)


@pytest.fixture
def torch_stacked_lstm():
    class LSTMClassifier(torch.nn.Module):
        """A simple LSTM for text classification."""

        def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
            self.lstm_layer = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=3)
            self.linear_layer_1 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.relu_layer = torch.nn.ReLU()
            self.linear_layer_2 = torch.nn.Linear(hidden_dim, target_size)
            self.classification_layer = torch.nn.Softmax(1)

        def forward(self, input_):
            embedding = self.embedding_layer(input_.long())
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
def torch_bidirectional_lstm():
    class LastCell(torch.nn.Module):
        def forward(self, x):
            out, _ = x
            return out[:, -1, :]

    return torch.nn.Sequential(
        torch.nn.Embedding(num_embeddings=5000, embedding_dim=64),
        torch.nn.LSTM(64, 64, bidirectional=True, batch_first=True),
        LastCell(),
        torch.nn.Linear(in_features=2 * 64, out_features=32),
    )


@pytest.fixture(
    params=[
        'torch_dense_model',
        'torch_simple_cnn_model',
        'torch_vgg16_cnn_model',
        'torch_stacked_lstm',
        'torch_bidirectional_lstm',
    ]
)
def torch_model(request):
    return request.getfixturevalue(request.param)
