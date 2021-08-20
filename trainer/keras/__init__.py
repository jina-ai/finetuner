from typing import Union, Callable

import tensorflow as tf
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

        wrapped_model.compile(
            loss=[head_layer.loss_fn, head_layer.loss_fn],
            metrics=[head_layer.metric_fn, head_layer.metric_fn],
        )
        wrapped_model.summary()
        return wrapped_model

    def _get_data_loader(self, inputs, batch_size=256, shuffle=False):

        ds = get_dataset(datasets, self.arity)
        input_shape = self.base_model.input_shape[1:]

        return (
            tf.data.Dataset.from_generator(
                lambda: ds(inputs),
                output_signature=(
                    tuple(
                        tf.TensorSpec(shape=input_shape, dtype=tf.float64)
                        for _ in range(self.arity)
                    ),
                    tf.TensorSpec(shape=(), dtype=tf.float64),
                ),
            )
            .shuffle(buffer_size=4096)
            .batch(batch_size, drop_remainder=True)
        )

    def fit(
        self,
        train_data: Union[
            DocumentArrayLike,
            Callable[..., DocumentArrayLike],
        ],
        batch_size: int = 256,
        **kwargs,
    ) -> None:
        self.wrapped_model.fit(
            self._get_data_loader(train_data),
            **kwargs,
        )

    def save(self, *args, **kwargs):
        self.base_model.save(*args, **kwargs)
