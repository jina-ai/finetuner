import abc
from typing import (
    Optional,
)

from ..helper import AnyDNN, EmbeddingLayerInfoType


class BaseTailor(abc.ABC):
    def __init__(
        self,
        model: AnyDNN,
        *args,
        **kwargs,
    ):
        """Tailor converts a general DNN model into an embedding model.

        :param model: a general DNN model
        :param args:
        :param kwargs:
        """
        self._model = model

    @abc.abstractmethod
    def convert(
        self,
        embedding_layer_name: Optional[str] = None,
        output_dim: Optional[int] = None,
        freeze: bool = False,
    ) -> AnyDNN:
        """Convert a general model from :py:attr:`.model` to an embedding model.

        :param embedding_layer_name: the name of the layer that is used for output embeddings. All layers *after* that layer
            will be removed. When set to ``None``, then the last layer listed in :py:attr:`.embedding_layers` will be used.
        :param output_dim: the dimensionality of the embedding output.
        :param freeze: if set, then freeze the weights in :py:attr:`.model`.

        """
        ...

    @property
    @abc.abstractmethod
    def embedding_layers(self) -> EmbeddingLayerInfoType:
        """Get all dense layers that can be used as embedding layer from the :py:attr:`.model`.

        :return: layers info as :class:`list` of :class:`dict`.
        """
        ...
