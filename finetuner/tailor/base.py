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
        """Freeze the weights of the DNN model.

        :return: if :attr:`self._freeze`, return a new model with all layers weights freezed.
        """
        ...

    @abc.abstractmethod
    def _trim(self) -> AnyDNN:
        """Trim an arbitrary Keras model to a embedding model.

        :return: A trimmed model based on the :attr:`self._layer_idx`. All layers
          include & after :attr:`self._layer_idx`will be chop-off.
        """
        ...

    @property
    @abc.abstractmethod
    def candidate_layers(self) -> CandidateLayerInfo:
        """Get all dense layers that can be used as embedding layer from the given model.

        :return: Candidate layers info as list of dictionary.
        """
        ...

    @property
    def model(self) -> AnyDNN:
        """Get the DNN model.

        :return: The parsed DNN model.
        """
        return self._model

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        ...
