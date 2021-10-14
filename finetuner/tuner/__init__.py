from typing import Optional, Dict

from jina.logging.predefined import default_logger

from ..helper import AnyDNN, DocumentArrayLike, get_framework, TunerReturnType


def _get_optimizer_kwargs(optimizer: str, custom_kwargs: Optional[Dict]):
    """Merges user-provided optimizer kwargs with default ones."""

    DEFAULT_OPTIMIZER_KWARGS = {
        'adam': {'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08},
        'rmsprop': {'rho': 0.99, 'momentum': 0.0, 'epsilon': 1e-08, 'centered': False},
        'sgd': {'momentum': 0.0, 'nesterov': False},
    }

    try:
        opt_kwargs = DEFAULT_OPTIMIZER_KWARGS[optimizer]
    except KeyError:
        raise ValueError(
            f'Optimizer "{optimizer}" not supported, the supported'
            ' optimizers are "adam", "rmsprop" and "sgd"'
        )

    # Raise warning for non-existing keys passed
    custom_kwargs = custom_kwargs or {}
    extra_args = set(custom_kwargs.keys()) - set(opt_kwargs.keys())
    if extra_args:
        default_logger.warning(
            f'The following arguments are not valid for the optimizer {optimizer}:'
            ' {extra_kwargs}'
        )

    # Update only existing keys
    opt_kwargs.update((k, v) for k, v in custom_kwargs.items() if k in opt_kwargs)

    return opt_kwargs


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
