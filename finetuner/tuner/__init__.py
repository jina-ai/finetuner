from typing import Optional, TYPE_CHECKING, Type, Dict

from ..helper import AnyDNN, DocumentArrayLike, get_framework

if TYPE_CHECKING:
    from .base import BaseTuner
    from .summary import Summary


def _get_tuner_class(dnn_model: AnyDNN) -> Type['BaseTuner']:
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
    eval_data: Optional[DocumentArrayLike] = None,
    epochs: int = 10,
    batch_size: int = 256,
    loss: str = 'CosineSiameseLoss',
    learning_rate: float = 1e-3,
    optimizer: str = 'adam',
    optimizer_kwargs: Optional[Dict] = None,
    device: str = 'cpu',
    **kwargs,
) -> 'Summary':
    """Finetune the model on the training data.

    :param embed_model: an embedding model
    :param train_data: Data on which to train the model
    :param eval_data: Data on which to evaluate the model at the end of each epoch
    :param epochs: Number of epochs to train the model
    :param batch_size: The batch size to use for training and evaluation
    :param loss: Which loss to use in training. Supported
        losses are:
        - ``CosineSiameseLoss`` for Siamese network with cosine distance
        - ``EuclideanSiameseLoss`` for Siamese network with eculidean distance
        - ``CosineTripletLoss`` for Triplet network with cosine distance
        - ``EuclideanTripletLoss`` for Triplet network with eculidean distance
    :param learning_rate: Learning rate to use in training
    :param optimizer: Which optimizer to use in training. Supported
        values/optimizers are:
        - ``"adam"`` for the Adam optimizer
        - ``"rmsprop"`` for the RMSProp optimizer
        - ``"sgd"`` for the SGD optimizer with momentum
    :param optimizer_kwargs: Keyword arguments to pass to the optimizer. The
        supported arguments, togethere with their defailt values, are:
        - ``"adam"``:  ``{'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08}``
        - ``"rmsprop"``::

            {
                'rho': 0.99,
                'momentum': 0.0,
                'epsilon': 1e-08,
                'centered': False,
            }

        - ``"sgd"``: ``{'momentum': 0.0, 'nesterov': False}``
    :param device: The device to which to move the model. Supported options are
        ``"cpu"`` and ``"cuda"`` (for GPU)
    """
    ft = _get_tuner_class(embed_model)

    return ft(embed_model, loss=loss).fit(
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
    """Save the embedding model.

    :param embed_model: The embedding model to save
    :param model_path: Path to file/folder where to save the model
    :param args: Arguments to pass to framework-specific tuner's ``save`` method
    :param kwargs: Keyword arguments to pass to framework-specific tuner's ``save``
        method
    """
    ft = _get_tuner_class(embed_model)

    ft(embed_model).save(model_path, *args, **kwargs)
