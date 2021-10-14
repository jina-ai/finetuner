# do not change this line manually
# this is managed by git tag and updated on every release
# NOTE: this represents the NEXT release version
__version__ = '0.0.3'

__default_tag_key__ = 'finetuner'

# define the high-level API: fit()
from typing import Dict, Optional, overload, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .helper import AnyDNN, DocumentArrayLike, TunerReturnType


# fit interface generated from Tuner
@overload
def fit(
    model: 'AnyDNN',  #: must be an embedding model
    train_data: 'DocumentArrayLike',
    eval_data: Optional['DocumentArrayLike'] = None,
    epochs: int = 10,
    batch_size: int = 256,
    head_layer: str = 'CosineLayer',
    learning_rate: float = 0.01,
    optimizer: str = 'adam',
    optimizer_kwargs: Optional[Dict] = None,
    device: str = 'cpu',
) -> 'TunerReturnType':
    ...


# fit interface derived from Tailor + Tuner
@overload
def fit(
    model: 'AnyDNN',
    train_data: 'DocumentArrayLike',
    eval_data: Optional['DocumentArrayLike'] = None,
    epochs: int = 10,
    batch_size: int = 256,
    head_layer: str = 'CosineLayer',
    learning_rate: float = 0.01,
    optimizer: str = 'adam',
    optimizer_kwargs: Optional[Dict] = None,
    to_embedding_model: bool = True,  #: below are tailor args
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
    layer_name: Optional[str] = None,
    output_dim: Optional[int] = None,
    freeze: bool = False,
    device: str = 'cpu',
) -> 'TunerReturnType':
    ...


# fit interface from Labeler + Tuner
@overload
def fit(
    model: 'AnyDNN',  #: must be an embedding model
    train_data: 'DocumentArrayLike',
    interactive: bool = True,  #: below are labeler args
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    runtime_backend: str = 'thread',
    head_layer: str = 'CosineLayer',
    learning_rate: float = 0.01,
    optimizer: str = 'adam',
    optimizer_kwargs: Optional[Dict] = None,
    device: str = 'cpu',
) -> None:
    ...


# fit interface from Labeler + Tailor + Tuner
@overload
def fit(
    model: 'AnyDNN',
    train_data: 'DocumentArrayLike',
    interactive: bool = True,  #: below are labeler args
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    runtime_backend: str = 'thread',
    head_layer: str = 'CosineLayer',
    learning_rate: float = 0.01,
    optimizer: str = 'adam',
    optimizer_kwargs: Optional[Dict] = None,    
    to_embedding_model: bool = True,  #: below are tailor args
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
    layer_name: Optional[str] = None,
    output_dim: Optional[int] = None,
    freeze: bool = False,
    device: str = 'cpu',
) -> None:
    ...


def fit(
    model: 'AnyDNN', train_data: 'DocumentArrayLike', *args, **kwargs
) -> Optional['TunerReturnType']:
    if kwargs.get('to_embedding_model', False):
        from .tailor import to_embedding_model

        model = to_embedding_model(model, *args, **kwargs)

    if kwargs.get('interactive', False):
        from .labeler import fit

        return fit(model, train_data, *args, **kwargs)
    else:
        from .tuner import fit

        return fit(model, train_data, *args, **kwargs)
