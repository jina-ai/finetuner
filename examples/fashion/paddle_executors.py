import numpy as np
import paddle
from jina import Executor, DocumentArray, requests


class MyPaddleEncoder(Executor):
    """"""

    def __init__(self, model_path: str = './trained', **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = paddle.jit.load(model_path)
        self.embedding_model.eval()

    @requests
    def encode(self, docs: 'DocumentArray', **kwargs):
        """Encode the data using an SVD decomposition

        :param docs: input documents to update with an embedding
        :param kwargs: other keyword arguments
        """
        content = np.stack(docs.get_attributes('content'))
        content = content[:, :, :, 0].reshape(-1, 28, 28)

        content = paddle.to_tensor(content)
        embeds = self.embedding_model(content / 255.0)
        for doc, embed, cont in zip(docs, embeds.numpy(), content.numpy()):
            doc.embedding = embed
            doc.content = cont
            doc.convert_image_blob_to_uri(width=28, height=28)
            doc.pop('blob')
