# do not change this line manually
# this is managed by git tag and updated on every release
# NOTE: this represents the NEXT release version
__version__ = '0.2.0'

__default_tag_key__ = 'finetuner'

# define the high-level API: fit()
from typing import Callable, Optional, overload, TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from .helper import AnyDNN, AnyOptimizer, DocumentSequence
    from .tuner.summary import Summary


# fit interface generated from Tuner
@overload
def fit(
    model: 'AnyDNN',  #: must be an embedding model
    train_data: 'DocumentSequence',
    eval_data: Optional['DocumentSequence'] = None,
    preprocess_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    epochs: int = 10,
    batch_size: int = 256,
    num_items_per_class: int = 4,
    loss: Union[str, 'AnyDNN'] = 'SiameseLoss',
    optimizer: Optional['AnyOptimizer'] = None,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
) -> Tuple['AnyDNN', 'Summary']:
    ...


# fit interface derived from Tailor + Tuner
@overload
def fit(
    model: 'AnyDNN',
    train_data: 'DocumentSequence',
    eval_data: Optional['DocumentSequence'] = None,
    preprocess_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    epochs: int = 10,
    batch_size: int = 256,
    num_items_per_class: int = 4,
    loss: Union[str, 'AnyDNN'] = 'SiameseLoss',
    optimizer: Optional['AnyOptimizer'] = None,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    to_embedding_model: bool = True,  #: below are tailor args
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
    layer_name: Optional[str] = None,
    output_dim: Optional[int] = None,
    freeze: bool = False,
) -> Tuple['AnyDNN', 'Summary']:
    ...


# fit interface from Labeler + Tuner
@overload
def fit(
    model: 'AnyDNN',  #: must be an embedding model
    train_data: 'DocumentSequence',
    eval_data: Optional['DocumentSequence'] = None,
    preprocess_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    epochs: int = 10,
    batch_size: int = 256,
    num_items_per_class: int = 4,
    loss: Union[str, 'AnyDNN'] = 'SiameseLoss',
    optimizer: Optional['AnyOptimizer'] = None,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    interactive: bool = True,  #: below are labeler args
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    runtime_backend: str = 'thread',
) -> Tuple['AnyDNN', None]:
    ...


# fit interface from Labeler + Tailor + Tuner
@overload
def fit(
    model: 'AnyDNN',
    train_data: 'DocumentSequence',
    eval_data: Optional['DocumentSequence'] = None,
    preprocess_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    epochs: int = 10,
    batch_size: int = 256,
    num_items_per_class: int = 4,
    loss: Union[str, 'AnyDNN'] = 'SiameseLoss',
    optimizer: Optional['AnyOptimizer'] = None,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
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
) -> Tuple['AnyDNN', None]:
    ...


def fit(
    model: 'AnyDNN', train_data: 'DocumentSequence', *args, **kwargs
) -> Tuple['AnyDNN', Optional['Summary']]:
    if kwargs.get('to_embedding_model', False):
        from .tailor import to_embedding_model

        model = to_embedding_model(model, *args, **kwargs)

    if kwargs.get('interactive', False):
        from .labeler import fit

        return model, fit(model, train_data, *args, **kwargs)
    else:
        from .tuner import fit

        return model, fit(model, train_data, *args, **kwargs)


# level them up to the top-level
from .tuner import save
from .tailor import display
from .embedding import embed
