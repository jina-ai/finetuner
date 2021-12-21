import os
import pickle
from typing import TYPE_CHECKING, Optional, Union

import keras
import tensorflow as tf
from keras.engine.data_adapter import KerasSequenceAdapter
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from ... import __default_tag_key__
from ..base import BaseLoss, BaseTuner
from ..dataset import ClassDataset, SessionDataset
from ..state import TunerState
from . import losses
from .data import KerasDataSequence

if TYPE_CHECKING:
    from ...helper import CollateFnType, DocumentSequence, PreprocFnType


class KerasTuner(
    BaseTuner[Layer, KerasSequenceAdapter, Optimizer, LearningRateSchedule]
):
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
        num_workers: int = 0,
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

    def _move_model_to_device(self):
        """Move the model to device and set device"""
        # This does nothing as explicit device placement is not required in Keras

    def _default_configure_optimizer(self, model: Layer) -> Optimizer:
        """Get the default Adam optimizer"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        return optimizer

    def _train(self, data: KerasSequenceAdapter):
        """Train the model on given labeled data"""

        for idx, (inputs, labels) in enumerate(data.get_dataset()):

            # Set state variables
            self.state.batch_index = idx
            self.state.learning_rates['learning_rate'] = self._optimizer._decayed_lr(
                tf.float32
            ).numpy()

            self._trigger_callbacks('on_train_batch_begin')

            with tf.GradientTape() as tape:
                embeddings = self._embed_model(inputs)
                loss = self._loss(embeddings, labels)

            grads = tape.gradient(loss, self._embed_model.trainable_weights)
            self._optimizer.apply_gradients(
                zip(grads, self._embed_model.trainable_weights)
            )

            self.state.current_loss = loss.numpy()
            self._trigger_callbacks('on_train_batch_end')

        data.on_epoch_end()  # To re-create batches

    def _eval(self, data: KerasSequenceAdapter):
        """Evaluate the model on given labeled data"""

        for idx, (inputs, labels) in enumerate(data.get_dataset()):
            self.state.batch_index = idx
            self._trigger_callbacks('on_val_batch_begin')

            embeddings = self._embed_model(inputs)
            loss = self._loss(embeddings, labels)

            self.state.current_loss = loss.numpy()
            self._trigger_callbacks('on_val_batch_end')

        data.on_epoch_end()  # To re-create batches

    def _fit(
        self,
        train_data: 'DocumentSequence',
        eval_data: Optional['DocumentSequence'] = None,
        epochs: int = 10,
        batch_size: int = 256,
        num_items_per_class: Optional[int] = None,
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
        num_workers: int = 0,
    ):
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

        # Set state
        self.state = TunerState(num_epochs=epochs)
        self._trigger_callbacks('on_fit_begin')

        with get_device(self._device_name):
            for epoch in range(epochs):

                # Setting here as re-shuffling can change number of batches
                self.state.epoch = epoch
                self.state.num_batches_train = train_dl.get_size()
                self.state.batch_index = 0

                self._trigger_callbacks('on_epoch_begin')

                self._trigger_callbacks('on_train_epoch_begin')
                self._train(train_dl)
                self._trigger_callbacks('on_train_epoch_end')

                if eval_data:
                    self.state.num_batches_val = eval_dl.get_size()
                    self.state.batch_index = 0

                    self._trigger_callbacks('on_val_begin')
                    self._eval(eval_dl)
                    self._trigger_callbacks('on_val_end')

                self._trigger_callbacks('on_epoch_end')

                if self.stop_training:
                    break

            self._trigger_callbacks('on_fit_end')

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
