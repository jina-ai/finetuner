from typing import Union, Optional, Iterator, Callable

import tensorflow as tf
from jina import DocumentArray, Document
from jina.types.arrays.memmap import DocumentArrayMemmap
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from . import head_models
from ..base import BaseTrainer


class KerasTrainer(BaseTrainer):
    def __init__(
        self,
        base_model: Optional[Model] = None,
        arity: int = 2,
        head_model: Union[Layer, str, None] = 'HatLayer',
        loss: str = 'hinge',
        **kwargs,
    ):
        super().__init__(base_model, arity, head_model, loss, **kwargs)

    @property
    def base_model(self) -> Model:
        return self._base_model

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def head_model(self) -> Layer:
        return self._head_model

    @property
    def loss(self) -> str:
        return self._loss

    @property
    def wrapped_model(self) -> Model:
        if self.base_model is None:
            raise ValueError(f'base_model is not set')

        input_shape = self.base_model.input_shape[1:]
        input_values = [keras.Input(shape=input_shape) for _ in range(self.arity)]
        if self.arity == 2:
            # siamese structure
            head_layer = getattr(head_models, self._head_model)()(
                *(self._base_model(v) for v in input_values)
            )
            wrapped_model = Model(inputs=input_values, outputs=head_layer)
        elif self.arity == 3:
            # triplet structure
            raise NotImplementedError
        else:
            raise NotImplementedError

        wrapped_model.compile(loss=self.loss)
        wrapped_model.summary()
        return wrapped_model

    def _da_gen(self, doc_array):
        if callable(doc_array):
            doc_array = doc_array()
        for d in doc_array:
            for m in d.matches:
                yield (d.content, m.content), 1 if int(
                    m.tags['trainer']['label']
                ) == 1 else -1

    def _da_to_tf_generator(self, doc_array):
        input_shape = self.base_model.input_shape[1:]

        return tf.data.Dataset.from_generator(
            lambda: self._da_gen(doc_array),
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
        **kwargs,
    ) -> None:
        self.wrapped_model.fit(
            self._da_to_tf_generator(doc_array)
            .shuffle(buffer_size=4096)
            .batch(512, drop_remainder=True),
            **kwargs,
        )

    def save(self, *args, **kwargs):
        self.base_model.save(*args, **kwargs)
