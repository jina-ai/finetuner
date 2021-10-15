from typing import Optional, Dict

from ..helper import AnyDNN, DocumentArrayLike, get_framework, TunerReturnType


def _get_tuner_class(embed_model):
    f_type = get_framework(embed_model)

    if f_type == 'keras':
        from .keras import KerasTuner

        return KerasTuner
    elif f_type == 'torch':
        from .pytorch import PytorchTuner

        return PytorchTuner
    elif f_type == 'paddle':
        from .paddle import PaddleTuner

        return PaddleTuner
    else:
        raise ValueError('Could not identify backend framework of embed_model.')


def fit(
    embed_model: AnyDNN,
    train_data: DocumentArrayLike,
    eval_data: Optional[DocumentArrayLike] = None,
    epochs: int = 10,
    batch_size: int = 256,
    head_layer: str = 'CosineLayer',
    device: str = 'cpu',
    **kwargs
) -> TunerReturnType:
    ft = _get_tuner_class(embed_model)

    return ft(embed_model, head_layer=head_layer).fit(
        train_data, eval_data, epochs=epochs, batch_size=batch_size, device=device
    )


def save(embed_model, model_path):
    ft = _get_tuner_class(embed_model)

    ft(embed_model).save(model_path)
