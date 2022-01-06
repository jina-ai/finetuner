__version__ = '0.3.1'

__default_tag_key__ = 'finetuner_label'

# define the high-level API: fit()
import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union, overload

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

# level them up to the top-level
from .embedding import embed  # noqa: F401
from .tailor import display  # noqa: F401
from .tuner import save  # noqa: F401

if TYPE_CHECKING:
    from .helper import (
        AnyDNN,
        AnyOptimizer,
        AnyScheduler,
        CollateFnType,
        DocumentSequence,
        PreprocFnType,
    )
    from .tuner.callback import BaseCallback


# To make logging pretty - but most impotantly, play nice with progress bar
class _RichHandler(RichHandler):
    def render_message(self, record: logging.LogRecord, message: str):
        """Add logger name to log message"""
        return Text(f'[{record.name}] ') + super().render_message(record, message)


live_console = Console()  # Can be used by other rich components, e.g. progress bar
_logger = logging.getLogger('finetuner')
_logger.addHandler(_RichHandler(console=live_console))


# fit interface generated from Tuner
@overload
def fit(
    model: 'AnyDNN',  #: must be an embedding model
    train_data: 'DocumentSequence',
    eval_data: Optional['DocumentSequence'] = None,
    epochs: int = 10,
    batch_size: int = 256,
    loss: Union[str, 'AnyDNN'] = 'SiameseLoss',
    configure_optimizer: Optional[
        Callable[
            ['AnyDNN'], Union['AnyOptimizer', Tuple['AnyOptimizer', 'AnyScheduler']]
        ]
    ] = None,
    learning_rate: float = 1e-3,
    scheduler_step: str = 'batch',
    device: str = 'cpu',
    preprocess_fn: Optional['PreprocFnType'] = None,
    collate_fn: Optional['CollateFnType'] = None,
    num_items_per_class: Optional[int] = None,
    callbacks: Optional[List['BaseCallback']] = None,
    num_workers: int = 0,
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
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
    configure_optimizer: Optional[
        Callable[
            ['AnyDNN'], Union['AnyOptimizer', Tuple['AnyOptimizer', 'AnyScheduler']]
        ]
    ] = None,
    learning_rate: float = 1e-3,
    scheduler_step: str = 'batch',
    device: str = 'cpu',
    preprocess_fn: Optional['PreprocFnType'] = None,
    collate_fn: Optional['CollateFnType'] = None,
    num_items_per_class: Optional[int] = None,
    callbacks: Optional[List['BaseCallback']] = None,
    num_workers: int = 0,
    to_embedding_model: bool = True,  #: below are tailor args
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
    layer_name: Optional[str] = None,
    freeze: Union[bool, List[str]] = False,
    bottleneck_net: Optional['AnyDNN'] = None,
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
    configure_optimizer: Optional[
        Callable[
            ['AnyDNN'], Union['AnyOptimizer', Tuple['AnyOptimizer', 'AnyScheduler']]
        ]
    ] = None,
    learning_rate: float = 1e-3,
    scheduler_step: str = 'batch',
    device: str = 'cpu',
    preprocess_fn: Optional['PreprocFnType'] = None,
    collate_fn: Optional['CollateFnType'] = None,
    num_items_per_class: Optional[int] = None,
    callbacks: Optional[List['BaseCallback']] = None,
    num_workers: int = 0,
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
    configure_optimizer: Optional[
        Callable[
            ['AnyDNN'], Union['AnyOptimizer', Tuple['AnyOptimizer', 'AnyScheduler']]
        ]
    ] = None,
    learning_rate: float = 1e-3,
    scheduler_step: str = 'batch',
    device: str = 'cpu',
    preprocess_fn: Optional['PreprocFnType'] = None,
    collate_fn: Optional['CollateFnType'] = None,
    num_items_per_class: Optional[int] = None,
    callbacks: Optional[List['BaseCallback']] = None,
    num_workers: int = 0,
    interactive: bool = True,  #: below are labeler args
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    runtime_backend: str = 'thread',
    to_embedding_model: bool = True,  #: below are tailor args
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
    layer_name: Optional[str] = None,
    freeze: Union[bool, List[str]] = False,
    bottleneck_net: Optional['AnyDNN'] = None,
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
