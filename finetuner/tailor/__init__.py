from .. import AnyDNN
from ..helper import get_framework


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

    _ft = ft(model, **kwargs)
    return _ft.model
