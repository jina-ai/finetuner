import abc
from typing import (
    Optional,
)

from ..helper import AnyDNN, EmbeddingLayerInfoType


class BaseTailor(abc.ABC):
    def __init__(
        self,
        model: AnyDNN,
        freeze: bool = False,
        embedding_layer_name: Optional[str] = None,
        output_dim: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """Tailor converts a general DNN model into an embedding model.

        :param model: a general DNN model
        :param freeze: if set, then freeze the weights in :py:attr:`.model`
        :param embedding_layer_name: the name of the layer that is used for output embeddings. All layers after that layer
            will be removed. When not given, then the last layer listed in :py:attr:`.embedding_layers` will be used.
        :param args:
        :param kwargs:
        """
        self._model = model
        self._freeze = freeze
        self._embedding_layer_name = embedding_layer_name
        self._output_dim = output_dim

    @abc.abstractmethod
    def _freeze_weights(self) -> 'BaseTailor':
        """Freeze the weights of :py:attr:`.model`."""
        ...

    @abc.abstractmethod
    def _trim(self) -> 'BaseTailor':
        """Trim :py:attr:`.model` to an embedding model."""
        ...

    @property
    @abc.abstractmethod
    def embedding_layers(self) -> EmbeddingLayerInfoType:
        """Get all dense layers that can be used as embedding layer from the :py:attr:`.model`.

        :return: layers info as :class:`list` of :class:`dict`.
        """
        ...

    @property
    def model(self) -> AnyDNN:
        """Get the DNN model of this object.

        :return: The DNN model.
        """
        return self._model

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Get the user-defined output dimensionality.

        :return: Output dimension of the attached linear layer
        """
        ...

    @output_dim.setter
    def output_dim(self, dim: int):
        """Set a new output dimension for the model.

        if set, the :py:attr:`self.model`'s attached dense layer will have this dim.
        :param dim: Dimensionality of the attached linear layer.
        """
        self._output_dim = dim

    @abc.abstractmethod
    def _attach_dense_layer(self):
        """Attach a dense layer to the end of the parsed model.

        .. note::
           The attached dense layer have the same shape as the last layer
           in the parsed model.
           The attached dense layer will ignore the :py:attr:`freeze`, this
           layer always trainable.
        """
        ...

    def __call__(self, *args, **kwargs):
        if self._freeze:
            self._trim()._freeze_weights()._attach_dense_layer()
        else:
            self._trim()._attach_dense_layer()
