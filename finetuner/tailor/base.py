import abc
from typing import Optional, Tuple, TYPE_CHECKING, Union, List

if TYPE_CHECKING:
    from ..helper import AnyDNN, LayerInfoType


class BaseTailor(abc.ABC):
    def __init__(
        self,
        model: 'AnyDNN',
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
        freeze: Union[bool, List[str]] = False,
        bottleneck_net: Optional['AnyDNN'] = None,
    ) -> 'AnyDNN':
        """Convert a general model from :py:attr:`.model` to an embedding model.

        :param layer_name: the name of the layer that is used for output embeddings. All layers *after* that layer
            will be removed. When set to ``None``, then the last layer listed in :py:attr:`.embedding_layers` will be used.
            To see all available names you can check ``name`` field of :py:attr:`.embedding_layers`.
        :param freeze: if set as True, will freeze all layers before :py:`attr`:`layer_name`. If set as list of str, will freeze layers by names.
        :param bottleneck_net: Attach a bottleneck net at the end of model, this module should always trainable.
        :return: Converted embedding model.
        """
        ...

    @property
    def embedding_layers(self) -> 'LayerInfoType':
        """Get all dense layers that can be used as embedding layer from the :py:attr:`.model`.

        :return: layers info as Dict.
        """
        _layers = self.summary()
        return [_l for _l in _layers if _l['is_embedding_layer']]

    @abc.abstractmethod
    def summary(self, include_identity_layer: bool = False) -> 'LayerInfoType':
        """The summary of the model architecture. To list all potential embedding layers, use :py:attr:`.embedding_layers`.

        :param include_identity_layer: if set, then identity layers are included and returned.
        :return: all layers info as Dict.
        """
        ...

    def display(self, *args, **kwargs) -> None:
        """Display the model architecture from :py:attr:`.summary` in a table.

        :param args: args pass to :py:attr:`.summary`
        :param kwargs: kwargs pass to :py:attr:`.summary`
        """
        from rich.table import Table
        from rich import print, box

        _summary = self.summary(*args, **kwargs)
        table = Table(box=box.SIMPLE)
        cols = ['name', 'output_shape_display', 'nb_params', 'trainable']
        for k in cols:
            table.add_column(k)
        for s in _summary:
            style = None
            if s['trainable']:
                style = 'bright_green'
            elif not s['trainable']:
                style = 'cyan'
            if 'identity' in s['name']:
                style = 'bright_black'
            table.add_row(*map(str, (s[v] for v in cols)), style=style)
        print(
            table,
            '[green]Green[/green] layers are trainable layers, '
            '[cyan]Cyan[/cyan] layers are non-trainable layers or frozen layers.\n'
            '[bright_black]Gray[/bright_black] layers indicates this layer has been replaced by an Identity layer.\n'
            'Use to_embedding_model(...) to create embedding model.',
        )
