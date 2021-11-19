from typing import Callable, Optional, Union, List, TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from jina import DocumentArray, DocumentArrayMemmap
    from jina.types.document import DocumentContentType
    from .helper import AnyDNN, AnyTensor

from .helper import get_framework

T = TypeVar('T')


def embed(
    docs: Union['DocumentArray', 'DocumentArrayMemmap'],
    embed_model: 'AnyDNN',
    device: str = 'cpu',
    batch_size: int = 256,
    preprocess_fn: Optional[Callable[['DocumentContentType'], T]] = None,
    collate_fn: Optional[Callable[[List[T]], 'AnyTensor']] = None,
) -> None:
    """Fill the embedding of Documents inplace by using `embed_model`

    :param docs: the Documents to be embedded
    :param embed_model: the embedding model written in Keras/Pytorch/Paddle
    :param device: the computational device for `embed_model`, can be either
        `cpu` or `cuda`.
    :param batch_size: number of Documents in a batch for embedding
    :param preprocess_fn: A pre-processing function. It should take as input the
        content of an item in the dataset and return the pre-processed content
    :param collate_fn: The collation function to merge the content of individual
        items into a batch. Should accept a list with the content of each item,
        and output a tensor (or a list/dict of tensors) that feed directly into the
        embedding model
    """

    if not preprocess_fn:
        preprocess_fn = lambda x: x  # noqa

    fm = get_framework(embed_model)
    globals()[f'_set_embeddings_{fm}'](
        docs, embed_model, device, batch_size, preprocess_fn, collate_fn
    )


def _set_embeddings_keras(
    docs: Union['DocumentArray', 'DocumentArrayMemmap'],
    embed_model: 'AnyDNN',
    device: str = 'cpu',
    batch_size: int = 256,
    preprocess_fn: Optional[Callable[['DocumentContentType'], T]] = None,
    collate_fn: Optional[Callable[[List[T]], 'AnyTensor']] = None,
):
    from .tuner.keras import get_device

    if not collate_fn:
        collate_fn = lambda x: np.array(x)  # noqa

    device = get_device(device)
    with device:
        for b in docs.batch(batch_size):
            inputs = [preprocess_fn(x.content) for x in b]
            batch_inputs = collate_fn(inputs)

            b.embeddings = embed_model(batch_inputs, training=False).numpy()


def _set_embeddings_torch(
    docs: Union['DocumentArray', 'DocumentArrayMemmap'],
    embed_model: 'AnyDNN',
    device: str = 'cpu',
    batch_size: int = 256,
    preprocess_fn: Optional[Callable[['DocumentContentType'], T]] = None,
    collate_fn: Optional[Callable[[List[T]], 'AnyTensor']] = None,
):
    from .tuner.pytorch import get_device

    device = get_device(device)

    import torch

    if not collate_fn:
        collate_fn = lambda x: torch.tensor(x, device=device)  # noqa

    embed_model = embed_model.to(device)
    is_training_before = embed_model.training
    embed_model.eval()
    with torch.inference_mode():
        for b in docs.batch(batch_size):
            inputs = [preprocess_fn(x.content) for x in b]
            batch_inputs = collate_fn(inputs).to(device)

            b.embeddings = embed_model(batch_inputs).cpu().detach().numpy()
    if is_training_before:
        embed_model.train()


def _set_embeddings_paddle(
    docs,
    embed_model,
    device: str = 'cpu',
    batch_size: int = 256,
    preprocess_fn: Optional[Callable[['DocumentContentType'], T]] = None,
    collate_fn: Optional[Callable[[List[T]], 'AnyTensor']] = None,
):
    from .tuner.paddle import get_device

    device = get_device(device)

    import paddle

    if not collate_fn:
        collate_fn = lambda x: paddle.to_tensor(inputs, place=device)  # noqa

    is_training_before = embed_model.training
    embed_model.to(device=device)
    embed_model.eval()
    for b in docs.batch(batch_size):
        inputs = [preprocess_fn(x.content) for x in b]
        batch_inputs = paddle.to_tensor(collate_fn(inputs), place=device)
        b.embeddings = embed_model(batch_inputs).numpy()
    if is_training_before:
        embed_model.train()
