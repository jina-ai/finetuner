from typing import Optional

from ..helper import get_framework, AnyDNN, DocumentArrayLike


def fit(
    embed_model: AnyDNN,
    head_layer: str,
    train_data: DocumentArrayLike,
    eval_data: Optional[DocumentArrayLike] = None,
    epochs: int = 10,
    batch_size: int = 256,
    **kwargs
):
    f = get_framework(embed_model)

    if f == 'keras':
        from .keras import KerasTuner

        ft = KerasTuner
    elif f == 'torch':
        from .pytorch import PytorchTuner

        ft = PytorchTuner
    elif f == 'paddle':
        from .paddle import PaddleTuner

        ft = PaddleTuner

    f = ft(embed_model, head_layer)
    return f.fit(train_data, eval_data, epochs=epochs, batch_size=batch_size)
