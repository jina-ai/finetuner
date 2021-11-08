from typing import Callable, Optional, Type, TYPE_CHECKING, Union

from ..helper import AnyDNN, AnyOptimizer, DocumentSequence, get_framework
from .base import BaseLoss, BaseMiner

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
    train_data: DocumentSequence,
    eval_data: Optional[DocumentSequence] = None,
    preprocess_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    epochs: int = 10,
    miner: Optional[BaseMiner] = None,
    batch_size: int = 256,
    num_items_per_class: int = 4,
    loss: Union[str, BaseLoss] = 'SiameseLoss',
    optimizer: Optional[AnyOptimizer] = None,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    **kwargs,
) -> 'Summary':
    """Finetune the model on the training data.

    :param embed_model: an embedding model
    :param train_data: Data on which to train the model
    :param eval_data: Data on which to evaluate the model at the end of each epoch
    :param preprocess_fn: A pre-processing function. It should take as input the
        content of an item in the dataset and return the pre-processed content
    :param collate_fn: The collation function to merge individual items into batch
        inputs for the model (plus labels). Will be passed to torch ``DataLoader``
    :param epochs: Number of epochs to train the model
    :param batch_size: The batch size to use for training and evaluation
    :param loss: Which loss to use in training. Supported
        losses are:
        - ``SiameseLoss`` for Siamese network
        - ``TripletLoss`` for Triplet network
    :param num_items_per_class: Number of items from a single class to include in
        the batch. Only relevant for class datasets
    :param learning_rate: Learning rate for the default optimizer. If you
        provide a custom optimizer, this learning rate will not apply.
    :param optimizer: The optimizer to use for training. If none is passed, an
        Adam optimizer is used by default, with learning rate specified by the
        ``learning_rate`` parameter.
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
        preprocess_fn=preprocess_fn,
        collate_fn=collate_fn,
        miner=miner,
        num_items_per_class=num_items_per_class,
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
