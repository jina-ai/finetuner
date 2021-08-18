from typing import Dict
import paddle
from paddle import nn
import numpy as np
from jina import Executor, DocumentArray, requests

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = paddle.nn.Flatten(start_axis=1)
        self.fc1 = paddle.nn.Linear(
            in_features=28 * 28, out_features=128, bias_attr=True
        )
        self.relu = paddle.nn.ReLU()
        self.fc2 = paddle.nn.Linear(in_features=128, out_features=32, bias_attr=True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class MyPaddleEncoder(Executor):
    """
    """

    def __init__(self, model_path: str = './trained', **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = SimpleNet()
        self.embedding_model.load_dict(paddle.load(model_path))
        self.embedding_model.eval()

    @requests
    def encode(self, docs: 'DocumentArray', **kwargs):
        """Encode the data using an SVD decomposition

        :param docs: input documents to update with an embedding
        :param kwargs: other keyword arguments
        """
        # reduce dimension to 50 by random orthogonal projection
        content = np.stack(docs.get_attributes('content'))
        content = content[:, :, :, 0].reshape(-1, 28, 28)

        content = paddle.to_tensor(content)
        embeds = self.embedding_model(content / 255.0)
        for doc, embed, cont in zip(docs, embeds.numpy(), content.numpy()):
            doc.embedding = embed
            doc.content = cont
            doc.convert_image_blob_to_uri(width=28, height=28)
            doc.pop('blob')
