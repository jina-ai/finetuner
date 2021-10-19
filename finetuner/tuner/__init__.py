from typing import Optional, TYPE_CHECKING, Type, Dict

from ..helper import AnyDNN, DocumentArrayLike, TunerReturnType, get_framework
from jina import DocumentArray

if TYPE_CHECKING:
    from .base import BaseTuner


def get_tuner_class(dnn_model: AnyDNN) -> Type['BaseTuner']:
    f_type = get_framework(dnn_model)

    if f_type == 'keras':
        from .keras import KerasTuner

        return KerasTuner
    elif f_type == 'torch':
        from .pytorch import PytorchTuner

        return PytorchTuner
    elif f_type == 'paddle':
        from .paddle import PaddleTuner

        return PaddleTuner


def fit(
    embed_model: AnyDNN,
    train_data: DocumentArrayLike,
    catalog: DocumentArrayLike = None,
    eval_data: Optional[DocumentArrayLike] = None,
    epochs: int = 10,
    batch_size: int = 256,
    loss: str = 'CosineSiameseLoss',
    learning_rate: float = 1e-3,
    optimizer: str = 'adam',
    optimizer_kwargs: Optional[Dict] = None,
    device: str = 'cpu',
    **kwargs,
) -> TunerReturnType:
    ft = get_tuner_class(embed_model)
    if catalog is None:
        train_data = DocumentArray(train_data() if callable(train_data) else train_data)
        catalog = DocumentArray()
        catalog.extend(train_data.traverse_flat(['r', 'm']))
        if eval_data is not None:
            eval_data = DocumentArray(eval_data() if callable(eval_data) else eval_data)
            catalog.extend(eval_data.traverse_flat(['r', 'm']))

    return ft(embed_model, catalog=catalog, loss=loss).fit(
        train_data,
        eval_data,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        learning_rate=learning_rate,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
    )


def save(embed_model: AnyDNN, model_path: str, *args, **kwargs) -> None:
    ft = get_tuner_class(embed_model)

    ft(embed_model).save(model_path, *args, **kwargs)
