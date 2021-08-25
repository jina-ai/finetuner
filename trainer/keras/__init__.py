from typing import Optional

import tensorflow as tf
from jina.logging.profile import ProgressBar
from tensorflow import keras
from tensorflow.keras import Model

from . import head_layers, datasets
from .head_layers import HeadLayer
from ..base import BaseTrainer, DocumentArrayLike
from ..helper import get_dataset


class KerasTrainer(BaseTrainer):
    @property
    def head_layer(self) -> HeadLayer:
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)
        elif isinstance(self._head_layer, HeadLayer):
            return self._head_layer

    @property
    def wrapped_model(self) -> Model:
        if self.base_model is None:
            raise ValueError(f'base_model is not set')

        input_shape = self.base_model.input_shape[1:]
        input_values = [keras.Input(shape=input_shape) for _ in range(self.arity)]
        head_layer = self.head_layer()
        head_values = head_layer(*(self.base_model(v) for v in input_values))
        wrapped_model = Model(inputs=input_values, outputs=head_values)

        return wrapped_model

    def _get_data_loader(self, inputs, batch_size: int, shuffle: bool):

        ds = get_dataset(datasets, self.arity)
        input_shape = self.base_model.input_shape[1:]

        tf_data = tf.data.Dataset.from_generator(
            lambda: ds(inputs),
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

        return tf_data.batch(batch_size, drop_remainder=True)

    def _train(self, data, model: Model, optimizer, pbar_description: str):
        head_layer = self.head_layer()

        losses = []
        metrics = []

        with ProgressBar(pbar_description) as p:
            for inputs, label in data:
                with tf.GradientTape() as tape:
                    outputs = model(inputs, training=True)
                    loss = head_layer.loss_fn(pred_val=outputs, target_val=label)
                    metric = head_layer.metric_fn(pred_val=outputs, target_val=label)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                losses.append(loss.numpy())
                metrics.append(metric.numpy())

                p.update(
                    details=f'Loss={sum(losses) / len(losses):.2f} Accuracy={sum(metrics) / len(metrics):.2f}'
                )

    def _eval(self, data, model: Model, pbar_description: str):
        head_layer = self.head_layer()

        losses = []
        metrics = []

        with ProgressBar(pbar_description) as p:
            for inputs, label in data:
                outputs = model(inputs, training=False)
                loss = head_layer.loss_fn(pred_val=outputs, target_val=label)
                metric = head_layer.metric_fn(pred_val=outputs, target_val=label)

                losses.append(loss.numpy())
                metrics.append(metric.numpy())

                p.update(
                    details=f'Loss={sum(losses) / len(losses):.2f} Accuracy={sum(metrics) / len(metrics):.2f}'
                )

    def fit(
        self,
        train_data: DocumentArrayLike,
        eval_data: Optional[DocumentArrayLike] = None,
        epochs: int = 10,
        batch_size: int = 256,
        **kwargs,
    ) -> None:

        model = self.wrapped_model

        _train_data = self._get_data_loader(
            inputs=train_data, batch_size=batch_size, shuffle=False
        )

        if eval_data:
            _eval_data = self._get_data_loader(
                inputs=eval_data, batch_size=batch_size, shuffle=False
            )

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

        for epoch in range(epochs):
            self._train(
                _train_data,
                model,
                optimizer,
                pbar_description=f'Epoch {epoch + 1}/{epochs}',
            )

            if eval_data:
                self._eval(_eval_data, model, pbar_description='Evaluating...')

    def save(self, *args, **kwargs):
        self.base_model.save(*args, **kwargs)
