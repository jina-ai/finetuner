from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union

from ..helper import get_framework

if TYPE_CHECKING:
    from ..helper import AnyDNN
    from .base import BaseTailor


def _get_tailor_class(dnn_model: 'AnyDNN') -> Type['BaseTailor']:
    f_type = get_framework(dnn_model)

    if f_type == 'keras':
        from .keras import KerasTailor

        return KerasTailor
    elif f_type == 'torch':
        from .pytorch import PytorchTailor

        return PytorchTailor
    elif f_type == 'paddle':
        from .paddle import PaddleTailor

        return PaddleTailor


def to_embedding_model(
    model: 'AnyDNN',
    layer_name: Optional[str] = None,
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
    freeze: Union[bool, List[str]] = False,
    projection_head: Optional['AnyDNN'] = None,
    **kwargs
) -> 'AnyDNN':
    """Convert a general model from :py:attr:`.model` to an embedding model.

    :param model: The DNN model to be converted.
    :param layer_name: the name of the layer that is used for output embeddings. All layers *after* that layer
        will be removed. When set to ``None``, then the last layer listed in :py:attr:`.embedding_layers` will be used.
        To see all available names you can check ``name`` field of :py:attr:`.embedding_layers`.
    :param input_size: The input size of the DNN model.
    :param input_dtype: The input data type of the DNN model.
    :param freeze: if set as True, will freeze all layers before :py:`attr`:`layer_name`. If set as list of str, will freeze layers by names.
    :param projection_head: Attach a module at the end of model, this module should be always trainable.
    """
    ft = _get_tailor_class(model)

    return ft(model, input_size, input_dtype).to_embedding_model(
        layer_name=layer_name,
        projection_head=projection_head,
        freeze=freeze,
    )


def display(
    model: 'AnyDNN',
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
) -> None:
    """Display the model architecture from :py:attr:`.summary` in a table.

    :param model: The DNN model to display.
    :param input_size: The input size of the DNN model.
    :param input_dtype: The input data type of the DNN model.
    """
    ft = _get_tailor_class(model)

    ft(model, input_size, input_dtype).display()
