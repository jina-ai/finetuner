from typing import Dict, Optional

import tensorflow as tf
from jina.logging.profile import ProgressBar
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer

from . import head_layers, datasets
from .head_layers import HeadLayer
from ..base import BaseTuner
from ...helper import DocumentArrayLike
from ..dataset.helper import get_dataset
from ..logger import LogGenerator


class KerasTuner(BaseTuner):
    @property
    def head_layer(self) -> HeadLayer:
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)
        elif isinstance(self._head_layer, HeadLayer):
            return self._head_layer

    @property
    def wrapped_model(self) -> Model:
        if self.embed_model is None:
            raise ValueError('embed_model is not set')

        if getattr(self, '_wrapped_model', None) is not None:
            return self._wrapped_model

        input_shape = self.embed_model.input_shape[1:]
        input_values = [keras.Input(shape=input_shape) for _ in range(self.arity)]
        head_layer = self.head_layer()
        head_values = head_layer(*(self.embed_model(v) for v in input_values))
        self._wrapped_model = Model(inputs=input_values, outputs=head_values)

        return self._wrapped_model

    def _get_data_loader(self, inputs, batch_size: int, shuffle: bool):

        ds = get_dataset(datasets, self.arity)
        input_shape = self.embed_model.input_shape[1:]

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
        head_layer = self.head_layer()

        losses = []
        metrics = []

        log_generator = LogGenerator('T', losses, metrics)

        with ProgressBar(
            description, message_on_done=log_generator, final_line_feed=False
        ) as p:

            for inputs, label in data:
                with tf.GradientTape() as tape:
                    outputs = self.wrapped_model(inputs, training=True)
                    loss = head_layer.loss_fn(pred_val=outputs, target_val=label)
                    metric = head_layer.metric_fn(pred_val=outputs, target_val=label)

                grads = tape.gradient(loss, self.wrapped_model.trainable_weights)
                optimizer.apply_gradients(
                    zip(grads, self.wrapped_model.trainable_weights)
                )

                losses.append(loss.numpy())
                metrics.append(metric.numpy())

                p.update(message=log_generator())

        return losses, metrics

    def _eval(self, data, description: str = 'Evaluating', train_log: str = ''):
        head_layer = self.head_layer()

        losses = []
        metrics = []

        log_generator = LogGenerator('E', losses, metrics, train_log)

        with ProgressBar(description, message_on_done=log_generator) as p:
            for inputs, label in data:
                outputs = self.wrapped_model(inputs, training=False)
                loss = head_layer.loss_fn(pred_val=outputs, target_val=label)
                metric = head_layer.metric_fn(pred_val=outputs, target_val=label)

                losses.append(loss.numpy())
                metrics.append(metric.numpy())

                p.update(message=log_generator())

        return losses, metrics

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
    ):

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
        device = tf.device(device)

        _optimizer = self._get_optimizer(optimizer, optimizer_kwargs, learning_rate)

        losses_train = []
        metrics_train = []
        losses_eval = []
        metrics_eval = []

        with device:
            for epoch in range(epochs):
                lt, mt = self._train(
                    _train_data,
                    _optimizer,
                    description=f'Epoch {epoch + 1}/{epochs}',
                )
                losses_train.extend(lt)
                metrics_train.extend(mt)

                if eval_data:
                    le, me = self._eval(
                        _eval_data, train_log=LogGenerator('T', lt, mt)()
                    )
                    losses_eval.extend(le)
                    metrics_eval.extend(me)

        return {
            'loss': {'train': losses_train, 'eval': losses_eval},
            'metric': {'train': metrics_train, 'eval': metrics_eval},
        }

    def save(self, *args, **kwargs):
        self.embed_model.save(*args, **kwargs)
