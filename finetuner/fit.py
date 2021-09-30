from typing import Optional, overload

from .helper import AnyDNN, DocumentArrayLike
from .tuner.fit import TunerReturnType


@overload
def fit(
    embed_model: AnyDNN,
    train_data: DocumentArrayLike,
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    runtime_backend: str = 'thread',
    interactive: bool = True,
    head_layer: str = 'CosineLayer',
) -> None:
    ...


@overload
def fit(
    embed_model: AnyDNN,
    train_data: DocumentArrayLike,
    eval_data: Optional[DocumentArrayLike] = None,
    epochs: int = 10,
    batch_size: int = 256,
    head_layer: str = 'CosineLayer',
) -> TunerReturnType:
    ...


def fit(*args, **kwargs) -> Optional[TunerReturnType]:
    if kwargs.get('interactive', False):
        kwargs.pop('interactive')
        from .labeler.fit import fit

        return fit(*args, **kwargs)
    else:
        from .tuner.fit import fit

        return fit(*args, **kwargs)
