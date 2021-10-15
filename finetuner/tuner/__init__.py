from typing import Optional, Dict, Any

from ..helper import AnyDNN, DocumentArrayLike, get_framework, TunerReturnType


def fit(
    embed_model: AnyDNN,
    train_data: DocumentArrayLike,
    eval_data: Optional[DocumentArrayLike] = None,
    epochs: int = 10,
    batch_size: int = 256,
    loss: str = 'CosineSiameseLoss',
    device: str = 'cpu',
    **kwargs
) -> TunerReturnType:
    f_type = get_framework(embed_model)

    if f_type == 'keras':
        from .keras import KerasTuner

        ft = KerasTuner
    elif f_type == 'torch':
        from .pytorch import PytorchTuner

        ft = PytorchTuner
    elif f_type == 'paddle':
        from .paddle import PaddleTuner

        ft = PaddleTuner

    return ft(embed_model, loss=loss).fit(
        train_data, eval_data, epochs=epochs, batch_size=batch_size, device=device
    )
