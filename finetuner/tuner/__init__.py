from typing import Optional, TYPE_CHECKING

from ..helper import AnyDNN, DocumentArrayLike, TunerReturnType

if TYPE_CHECKING:
    from .base import BaseTuner


def get_tuner_class(dnn_model: AnyDNN) -> 'BaseTuner':
    f_type = get_framework(dnn_model)

    if f_type == 'keras':
        from .tuner.keras import KerasTuner

        return KerasTuner
    elif f_type == 'torch':
        from .tuner.pytorch import PytorchTuner

        return PytorchTuner
    elif f_type == 'paddle':
        from .tuner.paddle import PaddleTuner

        return PaddleTuner


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
    ft = get_tuner_class(embed_model)

    return ft(embed_model, head_layer=head_layer).fit(
        train_data, eval_data, epochs=epochs, batch_size=batch_size, device=device
    )


def save(embed_model: AnyDNN, model_path: str, *args, **kwargs) -> None:
    ft = get_tuner_class(embed_model)

    ft(embed_model).save(model_path, *args, **kwargs)
