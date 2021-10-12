from typing import Optional, Tuple

from ..helper import get_framework, AnyDNN


def to_embedding_model(
    model: AnyDNN,
    layer_name: Optional[str] = None,
    output_dim: Optional[int] = None,
    freeze: bool = False,
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
    **kwargs
) -> AnyDNN:
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

    return ft(model, input_size, input_dtype).to_embedding_model(
        layer_name=layer_name, output_dim=output_dim, freeze=freeze
    )


def display(
    model: AnyDNN,
    input_size: Optional[Tuple[int, ...]] = None,
    input_dtype: str = 'float32',
) -> AnyDNN:
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

    return ft(model, input_size, input_dtype).display()
