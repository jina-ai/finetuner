from typing import Optional

from ..helper import get_framework, AnyDNN, DocumentArrayLike


def fit(
    embed_model: AnyDNN,
    head_layer: str,
    train_data: DocumentArrayLike,
    eval_data: Optional[DocumentArrayLike] = None,
    epochs: int = 10,
    batch_size: int = 256,
):
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

    return ft(embed_model, head_layer).fit(
        train_data, eval_data, epochs=epochs, batch_size=batch_size
    )
