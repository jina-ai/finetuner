from typing import Union, Optional, Iterator, Callable

import tensorflow as tf
from jina import DocumentArray, Document
from jina.types.arrays.memmap import DocumentArrayMemmap
from tensorflow import keras
from tensorflow.keras import Model

from . import head_layers
from .head_layers import HeadLayer
from ..base import BaseTrainer


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

    def _da_gen(self, doc_array, input_shape):
        if callable(doc_array):
            doc_array = doc_array()
        for d in doc_array:
            d_blob = d.blob.reshape(input_shape)
            for m in d.matches:
                yield (d_blob, m.blob.reshape(input_shape)), m.tags['trainer']['label']

    def _da_to_tf_generator(self, doc_array):
        input_shape = self.base_model.input_shape[1:]

        return tf.data.Dataset.from_generator(
            lambda: self._da_gen(doc_array, input_shape),
            output_signature=(
                tuple(
                    tf.TensorSpec(shape=input_shape, dtype=tf.float64)
                    for _ in range(self.arity)
                ),
                tf.TensorSpec(shape=(), dtype=tf.float64),
            ),
        )

    def fit(
        self,
        doc_array: Union[
            DocumentArray,
            DocumentArrayMemmap,
            Iterator[Document],
            Callable[..., Iterator[Document]],
        ],
        batch_size: int = 256,
        **kwargs,
    ) -> None:
        self.wrapped_model.fit(
            self._da_to_tf_generator(doc_array)
            .shuffle(buffer_size=4096)
            .batch(batch_size, drop_remainder=True),
            **kwargs,
        )

    def save(self, *args, **kwargs):
        self.base_model.save(*args, **kwargs)
