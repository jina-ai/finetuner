# do not change this line manually
# this is managed by git tag and updated on every release
# NOTE: this represents the NEXT release version
__version__ = '0.2.5'

__default_tag_key__ = 'finetuner_label'

# define the high-level API: fit()
from typing import List, Optional, overload, TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from .tuner.callback import BaseCallback
    from .helper import (
        AnyDNN,
        AnyOptimizer,
        DocumentSequence,
        PreprocFnType,
        CollateFnType,
    )


# fit interface generated from Tuner
@overload
def fit(
    model: 'AnyDNN',  #: must be an embedding model
    train_data: 'DocumentSequence',
    eval_data: Optional['DocumentSequence'] = None,
    epochs: int = 10,
    batch_size: int = 256,
    loss: Union[str, 'AnyDNN'] = 'SiameseLoss',
    optimizer: Optional['AnyOptimizer'] = None,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    preprocess_fn: Optional['PreprocFnType'] = None,
    collate_fn: Optional['CollateFnType'] = None,
    num_items_per_class: Optional[int] = None,
    callbacks: Optional[List['BaseCallback']] = None,
) -> 'AnyDNN':
    ...


# fit interface derived from Tailor + Tuner
@overload
def fit(
    model: 'AnyDNN',
    train_data: 'DocumentSequence',
    eval_data: Optional['DocumentSequence'] = None,
    epochs: int = 10,
    batch_size: int = 256,
    loss: Union[str, 'AnyDNN'] = 'SiameseLoss',
    optimizer: Optional['AnyOptimizer'] = None,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    preprocess_fn: Optional['PreprocFnType'] = None,
    collate_fn: Optional['CollateFnType'] = None,
    num_items_per_class: Optional[int] = None,
    callbacks: Optional[List['BaseCallback']] = None,
    to_embedding_model: bool = True,  #: below are tailor args
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
    layer_name: Optional[str] = None,
    output_dim: Optional[int] = None,
    freeze: bool = False,
    freeze_layers: Optional[List[str]] = None,
) -> 'AnyDNN':
    ...


# fit interface from Labeler + Tuner
@overload
def fit(
    model: 'AnyDNN',  #: must be an embedding model
    train_data: 'DocumentSequence',
    eval_data: Optional['DocumentSequence'] = None,
    epochs: int = 10,
    batch_size: int = 256,
    loss: Union[str, 'AnyDNN'] = 'SiameseLoss',
    optimizer: Optional['AnyOptimizer'] = None,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    preprocess_fn: Optional['PreprocFnType'] = None,
    collate_fn: Optional['CollateFnType'] = None,
    num_items_per_class: Optional[int] = None,
    callbacks: Optional[List['BaseCallback']] = None,
    interactive: bool = True,  #: below are labeler args
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    runtime_backend: str = 'thread',
) -> 'AnyDNN':
    ...


# fit interface from Labeler + Tailor + Tuner
@overload
def fit(
    model: 'AnyDNN',
    train_data: 'DocumentSequence',
    eval_data: Optional['DocumentSequence'] = None,
    epochs: int = 10,
    batch_size: int = 256,
    loss: Union[str, 'AnyDNN'] = 'SiameseLoss',
    optimizer: Optional['AnyOptimizer'] = None,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    preprocess_fn: Optional['PreprocFnType'] = None,
    collate_fn: Optional['CollateFnType'] = None,
    num_items_per_class: Optional[int] = None,
    callbacks: Optional[List['BaseCallback']] = None,
    interactive: bool = True,  #: below are labeler args
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    runtime_backend: str = 'thread',
    to_embedding_model: bool = True,  #: below are tailor args
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
    layer_name: Optional[str] = None,
    output_dim: Optional[int] = None,
    freeze: bool = False,
    freeze_layers: Optional[List[str]] = None,
) -> 'AnyDNN':
    ...


def fit(model: 'AnyDNN', train_data: 'DocumentSequence', *args, **kwargs) -> 'AnyDNN':
    if kwargs.get('to_embedding_model', False):
        from .tailor import to_embedding_model

        model = to_embedding_model(model, *args, **kwargs)

    if kwargs.get('interactive', False):
        from .labeler import fit

        fit(model, train_data, *args, **kwargs)
        return model
    else:
        from .tuner import fit

        fit(model, train_data, *args, **kwargs)
        return model


# level them up to the top-level
from .tuner import save
from .tailor import display
from .embedding import embed
