from typing import Optional, overload

from .helper import AnyDNN, DocumentArrayLike


@overload
def fit(
    embed_model: AnyDNN,
    train_data: DocumentArrayLike,
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    runtime_backend: str = 'thread',
    interactive: bool = True,
    head_layer: str = 'CosineLayer',
):
    ...


@overload
def fit(
    embed_model: AnyDNN,
    train_data: DocumentArrayLike,
    eval_data: Optional[DocumentArrayLike] = None,
    head_layer: str = 'CosineLayer',
    epochs: int = 10,
    batch_size: int = 256,
):
    ...


def fit(*args, **kwargs):
    if kwargs.get('interactive', False):
        kwargs.pop('interactive')
        from .labeler.fit import fit

        fit(*args, **kwargs)
    else:
        from .tuner.fit import fit

        fit(*args, **kwargs)
