from typing import overload, Optional, Tuple

from ..helper import get_framework, AnyDNN


# Keras Tailor
@overload
def convert(
    model: AnyDNN,
    freeze: bool = False,
    embedding_layer_name: Optional[str] = None,
    output_dim: Optional[int] = None,
) -> AnyDNN:
    ...


# Pytorch and Paddle Tailor
@overload
def convert(
    model: AnyDNN,
    input_size: Tuple[int, ...],
    input_dtype: str = 'float32',
    embedding_layer_name: Optional[str] = None,
    output_dim: Optional[int] = None,
    freeze: bool = False,
) -> AnyDNN:
    ...


def convert(model: AnyDNN, **kwargs) -> AnyDNN:
    f_type = get_framework(model)

    if f_type == 'keras':
        from .keras import KerasTailor

        ft = KerasTailor
    elif f_type == 'torch':
        from .pytorch import PytorchTailor

        ft = PytorchTailor
    elif f_type == 'paddle':
        from .paddle import PaddleTailor

        ft = PaddleTailor

    return ft(model, **kwargs).convert(**kwargs)
