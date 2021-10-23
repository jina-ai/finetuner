from .helper import DocumentArrayLike, AnyDNN, get_framework


def fill_embeddings(
    docs: DocumentArrayLike, embed_model: AnyDNN, device: str = 'cpu'
) -> None:
    """Fill the embedding of Documents inplace by using `embed_model`

    :param docs: the Documents to be embedded
    :param embed_model: the embedding model written in Keras/Pytorch/Paddle
    :param device: the computational device for `embed_model`, can be `cpu`, `cuda`, etc.

    """
    fm = get_framework(embed_model)
    vars()[f'fill_embeddings_{fm}'](docs, embed_model, device)


def fill_embeddings_keras(
    docs: DocumentArrayLike, embed_model: AnyDNN, device: str = 'cpu'
):
    from tuner.keras import get_device

    device = get_device(device)
    with device:
        embeddings = embed_model(docs.blobs)

    for doc, embed in zip(docs, embeddings):
        doc.embedding = embed.numpy()


def fill_embeddings_torch(
    docs: DocumentArrayLike, embed_model: AnyDNN, device: str = 'cpu'
):
    from tuner.pytorch import get_device

    device = get_device(device)

    import torch

    tensor = torch.tensor(docs.blobs, device=device)
    with torch.inference_mode():
        embeddings = embed_model(tensor)
        for doc, embed in zip(docs, embeddings):
            doc.embedding = embed.cpu().numpy()


def fill_embeddings_paddle(
    docs: DocumentArrayLike, embed_model: AnyDNN, device: str = 'cpu'
):
    from tuner.paddle import get_device

    get_device(device)

    import paddle

    embeddings = embed_model(paddle.Tensor(docs.blobs))
    for doc, embed in zip(docs, embeddings):
        doc.embedding = embed.item()
