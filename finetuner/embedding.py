from typing import Union

from jina import DocumentArray, DocumentArrayMemmap

from .helper import AnyDNN, get_framework


def embed(
    docs: Union[DocumentArray, DocumentArrayMemmap],
    embed_model: AnyDNN,
    device: str = 'cpu',
) -> None:
    """Fill the embedding of Documents inplace by using `embed_model`

    :param docs: the Documents to be embedded
    :param embed_model: the embedding model written in Keras/Pytorch/Paddle
    :param device: the computational device for `embed_model`, can be either `cpu` or `cuda`.

    """
    fm = get_framework(embed_model)
    globals()[f'_set_embeddings_{fm}'](docs, embed_model, device)


def _set_embeddings_keras(
    docs: Union[DocumentArray, DocumentArrayMemmap],
    embed_model: AnyDNN,
    device: str = 'cpu',
):
    from .tuner.keras import get_device

    device = get_device(device)
    with device:
        embeddings = embed_model(docs.blobs).numpy()

    docs.embeddings = embeddings


def _set_embeddings_torch(
    docs: Union[DocumentArray, DocumentArrayMemmap],
    embed_model: AnyDNN,
    device: str = 'cpu',
):
    from .tuner.pytorch import get_device

    device = get_device(device)

    import torch

    tensor = torch.tensor(docs.blobs, device=device)
    embed_model = embed_model.to(device)
    with torch.inference_mode():
        embeddings = embed_model(tensor).cpu().detach().numpy()

    docs.embeddings = embeddings


def _set_embeddings_paddle(
    docs: Union[DocumentArray, DocumentArrayMemmap],
    embed_model: AnyDNN,
    device: str = 'cpu',
):
    from .tuner.paddle import get_device

    get_device(device)

    import paddle

    embeddings = embed_model(paddle.Tensor(docs.blobs)).numpy()
    docs.embeddings = embeddings
