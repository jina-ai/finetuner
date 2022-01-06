import paddle
import pytest
import tensorflow as tf
import torch


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


@pytest.fixture
def paddle_dense_model():
    return paddle.nn.Sequential(
        paddle.nn.Linear(in_features=128, out_features=128),
        paddle.nn.ReLU(),
        paddle.nn.Linear(in_features=128, out_features=64),
        paddle.nn.ReLU(),
        paddle.nn.Linear(in_features=64, out_features=32),
        paddle.nn.ReLU(),
        paddle.nn.Linear(in_features=32, out_features=10),
        paddle.nn.Softmax(),
    )


@pytest.fixture
def paddle_simple_cnn_model():
    return paddle.nn.Sequential(
        paddle.nn.Conv2D(1, 32, 3, 1),
        paddle.nn.ReLU(),
        paddle.nn.Conv2D(32, 64, 3, 1),
        paddle.nn.ReLU(),
        paddle.nn.MaxPool2D(2),
        paddle.nn.Dropout(0.25),
        paddle.nn.Flatten(),
        paddle.nn.Linear(9216, 128),
        paddle.nn.Dropout(0.25),
        paddle.nn.Linear(128, 10),
        paddle.nn.Softmax(),
    )


@pytest.fixture
def paddle_vgg16_cnn_model():
    return paddle.vision.models.vgg16(pretrained=False)


@pytest.fixture
def paddle_stacked_lstm():
    class LSTMClassifier(paddle.nn.Layer):
        """A simple LSTM for text classification."""

        def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.embedding_layer = paddle.nn.Embedding(vocab_size, embedding_dim)
            self.lstm_layer = paddle.nn.LSTM(embedding_dim, hidden_dim, num_layers=3)
            self.linear_layer_1 = paddle.nn.Linear(hidden_dim, hidden_dim)
            self.relu_layer = paddle.nn.ReLU()
            self.linear_layer_2 = paddle.nn.Linear(hidden_dim, target_size)
            self.classification_layer = paddle.nn.Softmax(1)

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
def paddle_bidirectional_lstm():
    class LastCell(paddle.nn.Layer):
        def forward(self, x):
            out, _ = x
            return out[:, -1, :]

    return paddle.nn.Sequential(
        paddle.nn.Embedding(num_embeddings=5000, embedding_dim=64),
        paddle.nn.LSTM(64, 64, direction='bidirectional'),
        LastCell(),
        paddle.nn.Linear(in_features=128, out_features=32),
    )


@pytest.fixture(
    params=[
        'paddle_dense_model',
        'paddle_simple_cnn_model',
        'paddle_vgg16_cnn_model',
        'paddle_stacked_lstm',
        'paddle_bidirectional_lstm',
    ]
)
def paddle_model(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def tf_dense_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(128,)))  # (None, 128)
    model.add(tf.keras.layers.Dense(128, activation='relu'))  # (None, 128)
    model.add(tf.keras.layers.Dense(64, activation='relu'))  # (None, 64)
    model.add(tf.keras.layers.Dense(32, activation='relu'))  # (None, 32)
    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # (None, 10)
    return model


@pytest.fixture
def tf_simple_cnn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(32, 3, (1, 1), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, 3, (1, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


@pytest.fixture
def tf_vgg16_cnn_model():
    return tf.keras.applications.vgg16.VGG16(weights=None)


@pytest.fixture
def tf_stacked_lstm():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(1000, 1024, input_length=128))
    model.add(
        tf.keras.layers.LSTM(256, return_sequences=True)
    )  # this layer will not considered as candidate layer
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(
        tf.keras.layers.LSTM(256, return_sequences=False)
    )  # this layer will be considered as candidate layer
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    return model


@pytest.fixture
def tf_bidirectional_lstm():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(32),
        ]
    )


@pytest.fixture(
    params=[
        'tf_dense_model',
        'tf_simple_cnn_model',
        'tf_vgg16_cnn_model',
        'tf_stacked_lstm',
        'tf_bidirectional_lstm',
    ]
)
def tf_model(request):
    return request.getfixturevalue(request.param)
