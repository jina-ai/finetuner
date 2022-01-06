from typing import TYPE_CHECKING, Optional, Union

import tensorflow as tf
from keras.engine.data_adapter import KerasSequenceAdapter
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from ... import __default_tag_key__
from ...excepts import DimensionMismatchException
from ..base import BaseLoss, BaseTuner
from ..dataset import ClassDataset, SessionDataset
from ..dataset.datasets import InstanceDataset
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

    def _attach_projection_head(
        self,
        output_dim: Optional[int] = 128,
        num_layers: Optional[int] = 3,
    ):
        """Attach a projection head on top of the embed model for self-supervised learning.
        Calling this function will modify :attr:`self._embed_model` in place.

        :param output_dim: The output dimensionality of the projection, default 128, recommend 32, 64, 128, 256.
        :param num_layers: Number of layers of the projection head, default 3, recommend 2, 3.
        """
        # interpret embed model output shape
        rand_input = tf.constant(
            tf.random.uniform(self._input_size), dtype=self._input_dtype
        )
        print(f'\n\n{type(rand_input)}\n\n')
        output = self._embed_model(tf.expand_dims(rand_input, axis=0))
        if not len(output.shape) == 2:
            raise DimensionMismatchException(
                f'Expected input shape is 2d, got {len(output.shape)}.'
            )
        projection_head = _ProjectionHead(output.shape[1], output_dim, num_layers)
        embed_model_with_projection_head = tf.keras.Sequential()
        embed_model_with_projection_head.add(self._embed_model)
        embed_model_with_projection_head.add(projection_head)
        self._embed_model = embed_model_with_projection_head

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

        # If self-supervised, add projection head.
        if isinstance(train_dl.dataset, InstanceDataset):
            self._attach_projection_head(output_dim=128, num_layers=3)

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

            # If self-supervised, drop projection head
            if isinstance(train_dl.dataset, InstanceDataset):
                self._embed_model = tf.keras.Sequential(self._embed_model.layers[:-1])

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


class _ProjectionHead(tf.keras.layers.Layer):
    """Projection head used internally for self-supervised training.
    It is (by default) a simple 3-layer MLP to be attached on top of embedding model only for training purpose.
    After training, it should be cut-out from the embedding model.
    """

    EPSILON = 1e-5

    def __init__(self, in_features: int, output_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.head_layers = []
        is_last_layer = False
        for idx in range(num_layers):
            if idx == num_layers - 1:
                is_last_layer = True
            if not is_last_layer:
                self.head_layers.append(
                    tf.keras.layers.Dense(
                        units=in_features,
                        bias_initializer='zeros',
                    )
                )
                self.head_layers.append(
                    tf.keras.layers.BatchNormalization(epsilon=self.EPSILON)
                )
                self.head_layers.append(tf.keras.layers.ReLU())
            else:
                self.head_layers.append(
                    tf.keras.layers.Dense(
                        units=output_dim,
                        bias_initializer='zeros',
                    )
                )

    def call(self, x):
        for layer in self.head_layers:
            x = layer(x)
        return x
