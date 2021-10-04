import abc
from typing import (
    Optional,
)

from jina.logging.logger import JinaLogger

from ..helper import AnyDNN
from .helper import CandidateLayerInfo


class BaseTailor(abc.ABC):
    def __init__(
        self,
        model: Optional[AnyDNN] = None,
        layer_idx: int = -1,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        self._model = model
        self._freeze = freeze
        self._layer_idx = layer_idx
        self._logger = JinaLogger(self.__class__.__name__)

    @abc.abstractmethod
    def _freeze_weights(self) -> AnyDNN:
        ...

    @abc.abstractmethod
    def _trim(self) -> AnyDNN:
        ...

    @property
    @abc.abstractmethod
    def candidate_layers(self) -> CandidateLayerInfo:
        ...

    def __call__(self, *args, **kwargs):
        ...
