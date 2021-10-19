import abc
import warnings
from typing import (
    Optional,
    Union,
    Tuple,
    List,
    Dict,
)

from ..helper import AnyDNN, AnyDataLoader, AnyOptimizer, DocumentArrayLike
from .summary import SummaryCollection


class BaseLoss:
    arity: int


class BaseTuner(abc.ABC):
    def __init__(
        self,
        embed_model: Optional[AnyDNN] = None,
        loss: Union[AnyDNN, str] = 'CosineSiameseLoss',
        **kwargs,
    ):
        """Create the tuner instance.

        :param embed_model: Model that produces embeddings from inputs
        :param loss: Either the loss object instance, or the name of the loss function.
            Currently available losses are ``CosineSiameseLoss``,
            ``EuclideanSiameseLoss``, ``EuclideanTripletLoss`` and ``CosineTripletLoss``
        """
        self._embed_model = embed_model
        self._loss = self._get_loss(loss)
        self._train_data_len = 0
        self._eval_data_len = 0

    def _get_optimizer_kwargs(self, optimizer: str, custom_kwargs: Optional[Dict]):
        """Merges user-provided optimizer kwargs with default ones."""

        DEFAULT_OPTIMIZER_KWARGS = {
            'adam': {'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08},
            'rmsprop': {
                'rho': 0.99,
                'momentum': 0.0,
                'epsilon': 1e-08,
                'centered': False,
            },
            'sgd': {'momentum': 0.0, 'nesterov': False},
        }

        try:
            opt_kwargs = DEFAULT_OPTIMIZER_KWARGS[optimizer]
        except KeyError:
            raise ValueError(
                f'Optimizer "{optimizer}" not supported, the supported'
                ' optimizers are "adam", "rmsprop" and "sgd"'
            )

        # Raise warning for non-existing keys passed
        custom_kwargs = custom_kwargs or {}
        extra_args = set(custom_kwargs.keys()) - set(opt_kwargs.keys())
        if extra_args:
            warnings.warn(
                f'The following arguments are not valid for the optimizer {optimizer}:'
                f' {extra_args}'
            )

        # Update only existing keys
        opt_kwargs.update((k, v) for k, v in custom_kwargs.items() if k in opt_kwargs)

        return opt_kwargs

    @property
    def embed_model(self) -> AnyDNN:
        """Get the base model of this object."""
        return self._embed_model

    @property
    def arity(self) -> int:
        """Get the arity of this object.

        For example,
            - ``arity = 2`` corresponds to the siamese network;
            - ``arity = 3`` corresponds to the triplet network.
        """
        return self._loss.arity

    @abc.abstractmethod
    def _get_optimizer(
        self, optimizer: str, optimizer_kwargs: Optional[dict], learning_rate: float
    ) -> AnyOptimizer:
        """Get the optimizer for training."""

    @abc.abstractmethod
    def fit(
        self,
        train_data: DocumentArrayLike,
        eval_data: Optional[DocumentArrayLike] = None,
        epochs: int = 10,
        batch_size: int = 256,
        *args,
        **kwargs,
    ) -> SummaryCollection:
        """Fit the :py:attr:`.embed_model` on labeled data.

        Note that fitting changes the weights in :py:attr:`.embed_model` in-place. This allows one to consecutively
        call :py:func:`.fit` multiple times with different configs or data to get better models.
        """
        ...

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        """Save the weights of the :py:attr:`.embed_model`."""
        ...

    @abc.abstractmethod
    def _get_loss(self, loss: Union[str, BaseLoss]) -> BaseLoss:
        """Get the loss layer."""
        ...

    @abc.abstractmethod
    def _get_data_loader(
        self, inputs: DocumentArrayLike, batch_size: int, shuffle: bool
    ) -> AnyDataLoader:
        """Get framework specific data loader from the input data."""
        ...

    @abc.abstractmethod
    def _train(
        self, data: AnyDataLoader, optimizer, description: str
    ) -> Tuple[List, List]:
        """Train the model on given labeled data"""
        ...

    @abc.abstractmethod
    def _eval(
        self, data: AnyDataLoader, description: str = 'Evaluating'
    ) -> Tuple[List, List]:
        """Evaluate the model on given labeled data"""
        ...


class BaseDataset:
    def __init__(
        self,
        inputs: DocumentArrayLike,
    ):
        super().__init__()
        self._inputs = inputs() if callable(inputs) else inputs
