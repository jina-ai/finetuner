from typing import Union, Callable

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

from . import head_layers
from .head_layers import HeadLayer
from ..base import BaseTrainer, DocumentArrayLike


class KerasTrainer(BaseTrainer):
    @property
    def head_layer(self) -> HeadLayer:
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)()
        elif isinstance(self._head_layer, HeadLayer):
            return self._head_layer

    @property
    def wrapped_model(self) -> Model:
        if self.base_model is None:
            raise ValueError(f'base_model is not set')

        input_shape = self.base_model.input_shape[1:]
        input_values = [keras.Input(shape=input_shape) for _ in range(self.arity)]
        head_values = self.head_layer(*(self.base_model(v) for v in input_values))
        wrapped_model = Model(inputs=input_values, outputs=head_values)

        def metric_fn(y_true, y_predict):
            return tf.equal(tf.math.sign(y_true), tf.math.sign(y_predict))

        wrapped_model.compile(loss=self.loss, metrics=[metric_fn])
        wrapped_model.summary()
        return wrapped_model

    def _get_data_loader(self, inputs, batch_size=256, shuffle=False):
        if self.arity == 2:

            from ..dataset import SiameseMixin, Dataset

            class _SiameseDataset(SiameseMixin, Dataset):
                ...

            ds = _SiameseDataset
        elif self.arity == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError

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
