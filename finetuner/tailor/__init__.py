from typing import Optional, Tuple, TYPE_CHECKING, Type

from ..helper import get_framework, AnyDNN

if TYPE_CHECKING:
    from .base import BaseTailor


def get_tailor_class(dnn_model: AnyDNN) -> Type['BaseTailor']:
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
    model: AnyDNN,
    layer_name: Optional[str] = None,
    output_dim: Optional[int] = None,
    freeze: bool = False,
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
    **kwargs
) -> AnyDNN:
    ft = get_tailor_class(model)

    return ft(model, input_size, input_dtype).to_embedding_model(
        layer_name=layer_name, output_dim=output_dim, freeze=freeze
    )


def display(
    model: AnyDNN,
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
) -> AnyDNN:
    ft = get_tailor_class(model)

    return ft(model, input_size, input_dtype).display()
