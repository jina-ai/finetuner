from typing import Optional, Tuple, TYPE_CHECKING

from ..helper import get_framework, AnyDNN, get_tailor_class

if TYPE_CHECKING:
    from .base import BaseTailor


def get_tailor_class(dnn_model: AnyDNN) -> 'BaseTailor':
    f_type = get_framework(dnn_model)

    if f_type == 'keras':
        from .tailor.keras import KerasTailor

        return KerasTailor
    elif f_type == 'torch':
        from .tailor.pytorch import PytorchTailor

        return PytorchTailor
    elif f_type == 'paddle':
        from .tailor.paddle import PaddleTailor

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
