from typing import Optional

from ..helper import AnyDNN, DocumentArrayLike, TunerReturnType, get_tuner_class


def fit(
    embed_model: AnyDNN,
    train_data: DocumentArrayLike,
    eval_data: Optional[DocumentArrayLike] = None,
    epochs: int = 10,
    batch_size: int = 256,
    head_layer: str = 'CosineLayer',
    learning_rate: float = 1e-3,
    optimizer: str = 'adam',
    optimizer_kwargs: Optional[Dict] = None,
    device: str = 'cpu',
    **kwargs,
) -> TunerReturnType:
    ft = get_tuner_class(embed_model)

    return ft(embed_model, head_layer=head_layer).fit(
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
