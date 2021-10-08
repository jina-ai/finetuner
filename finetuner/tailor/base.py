import abc
from typing import (
    Optional,
    Tuple,
)

from ..helper import AnyDNN, EmbeddingLayerInfoType


class BaseTailor(abc.ABC):
    def __init__(
        self,
        model: AnyDNN,
        input_size: Optional[Tuple[int, ...]] = None,
        input_dtype: str = 'float32',
    ):
        """Tailor converts a general DNN model into an embedding model.

        :param model: a general DNN model
        :param input_size: a sequence of integers defining the shape of the input tensor. Note, batch size is *not* part
            of ``input_size``. It is required for :py:class:`PytorchTailor` and  :py:class:`PaddleTailor`, but not :py:class:`C`
        :param input_dtype: the data type of the input tensor.
        """
        self._model = model

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        self._input_size = input_size
        self._input_dtype = input_dtype

    @abc.abstractmethod
    def to_embedding_model(
        self,
        layer_name: Optional[str] = None,
        output_dim: Optional[int] = None,
        freeze: bool = False,
    ) -> AnyDNN:
        """Convert a general model from :py:attr:`.model` to an embedding model.

        :param layer_name: the name of the layer that is used for output embeddings. All layers *after* that layer
            will be removed. When set to ``None``, then the last layer listed in :py:attr:`.embedding_layers` will be used.
            To see all available names you can check ``name`` field of :py:attr:`.embedding_layers`.
        :param output_dim: the dimensionality of the embedding output.
        :param freeze: if set, then freeze all weights of the original model.

        """
        ...

    @property
    @abc.abstractmethod
    def embedding_layers(self) -> EmbeddingLayerInfoType:
        """Get all dense layers that can be used as embedding layer from the :py:attr:`.model`.

        :return: layers info as :class:`list` of :class:`dict`.
        """
        ...
