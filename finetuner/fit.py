from typing import Optional

from .tuner.base import AnyDNN, DocumentArrayLike


def fit(
    embed_model: AnyDNN,
    head_layer: str,
    train_data: DocumentArrayLike,
    eval_data: Optional[DocumentArrayLike] = None,
    epochs: int = 10,
    batch_size: int = 256,
):

    if 'keras.' in embed_model.__module__:
        from .tuner.keras import KerasTuner

        ft = KerasTuner
    elif 'torch.' in embed_model.__module__:
        from .tuner.pytorch import PytorchTuner

        ft = PytorchTuner
    elif 'paddle.' in embed_model.__module__:
        from .tuner.paddle import PaddleTuner

        ft = PaddleTuner
    else:
        raise ValueError(
            f'can not determine the backend from embed_model from {embed_model.__module__}'
        )

    f = ft(embed_model, head_layer)
    return f.fit(train_data, eval_data, epochs=epochs, batch_size=batch_size)
