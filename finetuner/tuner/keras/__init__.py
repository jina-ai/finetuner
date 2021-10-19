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
        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, BaseLoss):
            return loss

    def _get_data_loader(self, inputs, batch_size: int, shuffle: bool):

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
        self.embed_model.save(*args, **kwargs)
