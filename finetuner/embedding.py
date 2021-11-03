from typing import Union

from jina import DocumentArray, DocumentArrayMemmap

from .helper import AnyDNN, get_framework


def embed(
    docs: Union[DocumentArray, DocumentArrayMemmap],
    embed_model: AnyDNN,
    device: str = 'cpu',
    batch_size: int = 256,
) -> None:
    """Fill the embedding of Documents inplace by using `embed_model`

    :param docs: the Documents to be embedded
    :param embed_model: the embedding model written in Keras/Pytorch/Paddle
    :param device: the computational device for `embed_model`, can be either `cpu` or `cuda`.
    :param batch_size: number of Documents in a batch for embedding
    """
    fm = get_framework(embed_model)
    globals()[f'_set_embeddings_{fm}'](docs, embed_model, device, batch_size)


def _set_embeddings_keras(
    docs: Union[DocumentArray, DocumentArrayMemmap],
    embed_model: AnyDNN,
    device: str = 'cpu',
    batch_size: int = 256,
):
    from .tuner.keras import get_device

    device = get_device(device)
    with device:
        for b in docs.batch(batch_size):
            b.embeddings = embed_model(b.blobs).numpy()


def _set_embeddings_torch(
    docs: Union[DocumentArray, DocumentArrayMemmap],
    embed_model: AnyDNN,
    device: str = 'cpu',
    batch_size: int = 256,
):
    from .tuner.pytorch import get_device

    device = get_device(device)

    import torch

    embed_model = embed_model.to(device)
    with torch.inference_mode():
        for b in docs.batch(batch_size):
            tensor = torch.tensor(b.blobs, device=device)
            b.embeddings = embed_model(tensor).cpu().detach().numpy()


def _set_embeddings_paddle(
    docs: Union[DocumentArray, DocumentArrayMemmap],
    embed_model: AnyDNN,
    device: str = 'cpu',
    batch_size: int = 256,
):
    from .tuner.paddle import get_device

    get_device(device)

    import paddle

    for b in docs.batch(batch_size):
        b.embeddings = embed_model(paddle.Tensor(b.blobs)).numpy()
