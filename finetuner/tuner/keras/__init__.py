from typing import Dict, Optional, Union

import numpy as np
import tensorflow as tf
from jina.logging.profile import ProgressBar
from tensorflow import keras
from tensorflow.keras.optimizers import Optimizer

from . import losses, datasets
from ..base import BaseTuner, BaseLoss
from ..dataset.helper import get_dataset
from ..logger import LogGenerator
from ..stats import TunerStats
from ...helper import DocumentArrayLike


class KerasTuner(BaseTuner):
    def _get_loss(self, loss: Union[BaseLoss, str]):
        """Get the loss layer."""

        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, BaseLoss):
            return loss

    def _get_data_loader(self, inputs, batch_size: int, shuffle: bool):
        """Get tensorflow ``Dataset`` from the input data. """

        ds = get_dataset(datasets, self.arity)
        input_shape = self.embed_model.input_shape[1:]

        tf_data = tf.data.Dataset.from_generator(
            lambda: ds(inputs, self._catalog),
            output_signature=(
                tuple(
                    tf.TensorSpec(shape=input_shape, dtype=tf.float32)
                    for _ in range(self.arity)
                ),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        )

        if shuffle:
            tf_data = tf_data.shuffle(buffer_size=4096)

        return tf_data.batch(batch_size)

    def _get_optimizer(
        self, optimizer: str, optimizer_kwargs: Optional[dict], learning_rate: float
    ) -> Optimizer:
        """Get the optimizer for training."""

        optimizer_kwargs = self._get_optimizer_kwargs(optimizer, optimizer_kwargs)

        if optimizer == 'adam':
            return keras.optimizers.Adam(
                learning_rate=learning_rate, **optimizer_kwargs
            )
        elif optimizer == 'rmsprop':
            return keras.optimizers.RMSprop(
                learning_rate=learning_rate, **optimizer_kwargs
            )
        elif optimizer == 'sgd':
            return keras.optimizers.SGD(learning_rate=learning_rate, **optimizer_kwargs)

    def _train(self, data, optimizer, description: str):
        """Train the model on given labeled data"""

        losses = []

        log_generator = LogGenerator('T', losses)

        with ProgressBar(
            description,
            message_on_done=log_generator,
            final_line_feed=False,
            total_length=self._train_data_len,
        ) as p:
            self._train_data_len = 0
            for inputs, label in data:
                with tf.GradientTape() as tape:
                    embeddings = [self._embed_model(inpt) for inpt in inputs]
                    loss = self._loss([*embeddings, label])

                grads = tape.gradient(loss, self._embed_model.trainable_weights)
                optimizer.apply_gradients(
                    zip(grads, self._embed_model.trainable_weights)
                )

                losses.append(loss.numpy())

                p.update(message=log_generator())
                self._train_data_len += 1

        return losses

    def _eval(self, data, description: str = 'Evaluating', train_log: str = ''):
        """Evaluate the model on given labeled data"""

        losses = []

        log_generator = LogGenerator('E', losses, train_log)

        with ProgressBar(
            description, message_on_done=log_generator, total_length=self._eval_data_len
        ) as p:
            self._eval_data_len = 0
            for inputs, label in data:
                embeddings = [self._embed_model(inpt) for inpt in inputs]
                loss = self._loss([*embeddings, label])

                losses.append(loss.numpy())

                p.update(message=log_generator())
                self._eval_data_len += 1

        return losses

    def fit(
        self,
        train_data: DocumentArrayLike,
        eval_data: Optional[DocumentArrayLike] = None,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        optimizer: str = 'adam',
        optimizer_kwargs: Optional[Dict] = None,
        device: str = 'cpu',
        **kwargs,
    ) -> TunerStats:
        """Finetune the model on the training data.

        :param train_data: Data on which to train the model
        :param eval_data: Data on which to evaluate the model at the end of each epoch
        :param epoch: Number of epochs to train the model
        :param batch_size: The batch size to use for training and evaluation
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

        _train_data = self._get_data_loader(
            inputs=train_data, batch_size=batch_size, shuffle=False
        )

        if eval_data:
            _eval_data = self._get_data_loader(
                inputs=eval_data, batch_size=batch_size, shuffle=False
            )

        if device == 'cuda':
            device = '/GPU:0'
        elif device == 'cpu':
            device = '/CPU:0'
        else:
            raise ValueError(f'Device {device} not recognized')
        self.device = tf.device(device)

        _optimizer = self._get_optimizer(optimizer, optimizer_kwargs, learning_rate)

        stats = TunerStats()

        with self.device:
            for epoch in range(epochs):
                lt = self._train(
                    _train_data,
                    _optimizer,
                    description=f'Epoch {epoch + 1}/{epochs}',
                )
                stats.add_train_loss(lt)

                if eval_data:
                    le = self._eval(_eval_data, train_log=LogGenerator("T", lt)())
                    stats.add_eval_loss(le)
                    stats.add_eval_metric(self.get_metrics(eval_data))

                stats.print_last()
        return stats

    def get_embeddings(self, data: DocumentArrayLike):
        blobs = data.blobs
        with self.device:
            embeddings = self.embed_model(blobs)
        for doc, embed in zip(data, embeddings):
            doc.embedding = np.array(embed)

    def save(self, *args, **kwargs):
        """Save the embedding model.

        You need to pass the path where to save the model in either ``args`` or
        ``kwargs`` (for ``filepath`` key).

        :param args: Arguments to pass to ``save`` method of the embedding model
        :param kwargs: Keyword arguments to pass to ``save`` method of the embedding
            model
        """

        self.embed_model.save(*args, **kwargs)
