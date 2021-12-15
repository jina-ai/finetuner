from typing import Callable, List, Optional, Type, TYPE_CHECKING, Tuple, Union

from .base import BaseLoss
from ..helper import get_framework

if TYPE_CHECKING:
    from .base import BaseTuner
    from .callback import BaseCallback
    from ..helper import (
        AnyDNN,
        AnyOptimizer,
        AnyScheduler,
        DocumentSequence,
        PreprocFnType,
        CollateFnType,
    )


def _get_tuner_class(dnn_model: 'AnyDNN') -> Type['BaseTuner']:
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
    embed_model: 'AnyDNN',
    train_data: 'DocumentSequence',
    eval_data: Optional['DocumentSequence'] = None,
    preprocess_fn: Optional['PreprocFnType'] = None,
    collate_fn: Optional['CollateFnType'] = None,
    epochs: int = 10,
    batch_size: int = 256,
    num_items_per_class: Optional[int] = None,
    loss: Union[str, BaseLoss] = 'SiameseLoss',
    configure_optimizer: Optional[
        Callable[
            ['AnyDNN'], Union['AnyOptimizer', Tuple['AnyOptimizer', 'AnyScheduler']]
        ]
    ] = None,
    learning_rate: float = 1e-3,
    scheduler_step: str = 'batch',
    device: str = 'cpu',
    callbacks: Optional[List['BaseCallback']] = None,
    num_workers: int = 0,
    **kwargs,
):
    """Finetune the model on the training data.

    :param embed_model: an embedding model
    :param train_data: Data on which to train the model
    :param eval_data: Data on which to evaluate the model at the end of each epoch
    :param preprocess_fn: A pre-processing function, to apply pre-processing to
        documents on the fly. It should take as input the document in the dataset,
        and output whatever content the framework-specific dataloader (and model) would
        accept.
    :param collate_fn: The collation function to merge the content of individual
        items into a batch. Should accept a list with the content of each item,
        and output a tensor (or a list/dict of tensors) that feed directly into the
        embedding model
    :param epochs: Number of epochs to train the model
    :param batch_size: The batch size to use for training and evaluation
    :param loss: Which loss to use in training. Supported
        losses are:
        - ``SiameseLoss`` for Siamese network
        - ``TripletLoss`` for Triplet network
    :param num_items_per_class: Number of items from a single class to include in
        the batch. Only relevant for class datasets
    :param configure_optimizer: A function that allows you to provide a custom
        optimizer and learning rate. The function should take one input - the
        embedding model, and return either just an optimizer or a tuple of an
        optimizer and a learning rate scheduler.
    :param learning_rate: Learning rate for the default optimizer. If you
        provide a custom optimizer, this learning rate will not apply.
    :param scheduler_step: At which interval should the learning rate sheduler's
        step function be called. Valid options are "batch" and "epoch".
    :param device: The device to which to move the model. Supported options are
        ``"cpu"`` and ``"cuda"`` (for GPU)
    :param callbacks: A list of callbacks. The progress bar callback
        will be pre-prended to this list.
    :param num_workers: Number of workers used for loading the data.

        This works only with Pytorch and Paddle Paddle, and has no effect when using
        a Keras model.
    """
    ft = _get_tuner_class(embed_model)

    ft(
        embed_model,
        loss=loss,
        callbacks=callbacks,
        configure_optimizer=configure_optimizer,
        learning_rate=learning_rate,
        scheduler_step=scheduler_step,
        device=device,
    ).fit(
        train_data,
        eval_data,
        epochs=epochs,
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        collate_fn=collate_fn,
        num_items_per_class=num_items_per_class,
    )


def save(embed_model: 'AnyDNN', model_path: str, *args, **kwargs) -> None:
    """Save the embedding model.

    :param embed_model: The embedding model to save
    :param model_path: Path to file/folder where to save the model
    :param args: Arguments to pass to framework-specific tuner's ``save`` method
    :param kwargs: Keyword arguments to pass to framework-specific tuner's ``save``
        method
    """
    ft = _get_tuner_class(embed_model)

    ft(embed_model).save(model_path, *args, **kwargs)
