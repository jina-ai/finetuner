from typing import Optional, Union, TYPE_CHECKING

import tensorflow as tf
from jina.logging.profile import ProgressBar
from keras.engine.data_adapter import KerasSequenceAdapter
from tensorflow.keras.optimizers import Optimizer

from . import losses
from .data import KerasDataSequence
from ..base import BaseTuner, BaseLoss
from ..dataset import ClassDataset, SessionDataset
from ..summary import ScalarSequence, Summary
from ... import __default_tag_key__

if TYPE_CHECKING:
    from ...helper import DocumentSequence, PreprocFnType, CollateFnType


class KerasTuner(BaseTuner[tf.keras.layers.Layer, KerasSequenceAdapter, Optimizer]):
    def _get_loss(self, loss: Union[BaseLoss, str]) -> BaseLoss:
        """Get the loss layer."""
        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, BaseLoss):
            return loss

    def _get_data_loader(
        self,
        data: 'DocumentSequence',
        batch_size: int,
        shuffle: bool,
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
        num_items_per_class: Optional[int] = None,
    ) -> KerasSequenceAdapter:
        """Get the dataloader for the dataset

        In this case, since there is no true dataloader in keras, we are returning
        the adapter, which can produce the dataset that yields batches.
        """

        if __default_tag_key__ in data[0].tags:
            dataset = ClassDataset(data, preprocess_fn=preprocess_fn)
        else:
            dataset = SessionDataset(data, preprocess_fn=preprocess_fn)

        batch_sampler = self._get_batch_sampler(
            dataset,
            batch_size,
            shuffle=shuffle,
            num_items_per_class=num_items_per_class,
        )
        sequence = KerasDataSequence(
            dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn
        )

        adapter = KerasSequenceAdapter(sequence)
        return adapter

    def _get_default_optimizer(self, learning_rate: float) -> Optimizer:
        """Get the default optimizer (Adam), if none was provided by user."""

        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _train(
        self, data: KerasSequenceAdapter, optimizer: Optimizer, description: str
    ) -> ScalarSequence:
        """Train the model on given labeled data"""

        _summary = ScalarSequence('Train Loss')
        with ProgressBar(
            description,
            message_on_done=_summary.__str__,
            final_line_feed=False,
            total_length=data.get_size(),
        ) as p:
            for inputs, labels in data.get_dataset():
                with tf.GradientTape() as tape:
                    embeddings = self._embed_model(inputs)
                    loss = self._loss(embeddings, labels)

                grads = tape.gradient(loss, self._embed_model.trainable_weights)
                optimizer.apply_gradients(
                    zip(grads, self._embed_model.trainable_weights)
                )

                _summary += loss.numpy()

                p.update(message=str(_summary))

        data.on_epoch_end()  # To re-create batches
        return _summary

    def _eval(
        self,
        data: KerasSequenceAdapter,
        description: str = 'Evaluating',
        train_loss: Optional[ScalarSequence] = None,
    ) -> ScalarSequence:
        """Evaluate the model on given labeled data"""

        _summary = ScalarSequence('Eval Loss')

        with ProgressBar(
            description,
            message_on_done=lambda: f'{train_loss} | {_summary}',
            total_length=data.get_size(),
        ) as p:
            for inputs, labels in data.get_dataset():
                embeddings = self._embed_model(inputs)
                loss = self._loss(embeddings, labels)

                _summary += loss.numpy()

                p.update(message=str(_summary))

        data.on_epoch_end()  # To re-create batches
        return _summary

    def fit(
        self,
        train_data: 'DocumentSequence',
        eval_data: Optional['DocumentSequence'] = None,
        epochs: int = 10,
        batch_size: int = 256,
        num_items_per_class: Optional[int] = None,
        optimizer: Optional[Optimizer] = None,
        learning_rate: float = 1e-3,
        device: str = 'cpu',
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
        **kwargs,
    ) -> Summary:
        """Finetune the model on the training data.

        :param train_data: Data on which to train the model
        :param eval_data: Data on which to evaluate the model at the end of each epoch
        :param preprocess_fn: A pre-processing function. It should take as input the
            content of an item in the dataset and return the pre-processed content
        :param collate_fn: The collation function to merge the content of individual
            items into a batch. Should accept a list with the content of each item,
            and output a tensor (or a list/dict of tensors) that feed directly into the
            embedding model
        :param epochs: Number of epochs to train the model
        :param batch_size: The batch size to use for training and evaluation
        :param num_items_per_class: Number of items from a single class to include in
            the batch. Only relevant for class datasets
        :param optimizer: The optimizer to use for training. If none is passed, an
            Adam optimizer is used by default, with learning rate specified by the
            ``learning_rate`` parameter.
        :param learning_rate: Learning rate for the default optimizer. If you
            provide a custom optimizer, this learning rate will not apply.
        :param device: The device to which to move the model. Supported options are
            ``"cpu"`` and ``"cuda"`` (for GPU)
        """

        # Get dataloaders
        train_dl = self._get_data_loader(
            train_data,
            batch_size=batch_size,
            num_items_per_class=num_items_per_class,
            shuffle=True,
            preprocess_fn=preprocess_fn,
            collate_fn=collate_fn,
        )
        if eval_data:
            eval_dl = self._get_data_loader(
                eval_data,
                batch_size=batch_size,
                num_items_per_class=num_items_per_class,
                shuffle=False,
                preprocess_fn=preprocess_fn,
                collate_fn=collate_fn,
            )

        optimizer = optimizer or self._get_default_optimizer(learning_rate)

        m_train_loss = ScalarSequence('train')
        m_eval_loss = ScalarSequence('eval')
        with get_device(device):
            for epoch in range(epochs):
                lt = self._train(
                    train_dl,
                    optimizer,
                    description=f'Epoch {epoch + 1}/{epochs}',
                )
                m_train_loss += lt

                if eval_data:
                    le = self._eval(eval_dl, train_loss=m_train_loss)
                    m_eval_loss += le

        return Summary(m_train_loss, m_eval_loss)

    def save(self, *args, **kwargs):
        """Save the embedding model.

        You need to pass the path where to save the model in either ``args`` or
        ``kwargs`` (for ``filepath`` key).

        :param args: Arguments to pass to ``save`` method of the embedding model
        :param kwargs: Keyword arguments to pass to ``save`` method of the embedding
            model
        """

        self.embed_model.save(*args, **kwargs)


def get_device(device: str):
    """Get tensorflow compute device.

    :param device: device name
    """

    # translate our own alias into framework-compatible ones
    if device == 'cuda':
        device = '/GPU:0'
    elif device == 'cpu':
        device = '/CPU:0'
    return tf.device(device)
