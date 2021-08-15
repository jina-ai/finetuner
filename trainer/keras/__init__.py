from typing import Union, Optional

import tensorflow as tf
from jina import DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from ..base import BaseTrainer


class HatLayer(Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def call(self, lvalue, rvalue):
        return tf.concat([lvalue, rvalue, tf.abs(lvalue - rvalue)], axis=-1)


class KerasTrainer(BaseTrainer):
    def __init__(
            self,
            base_model: Optional[Model] = None,
            arity: int = 2,
            head_model: Union[Layer, str, None] = None,
            loss: str = 'hinge',
            **kwargs,
    ):
        self._base_model = base_model
        self._head_model = head_model
        self._arity = arity
        self._loss = loss

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

    def _compile(self) -> Model:
        if self.base_model is None:
            raise ValueError(f'base_model is not set')

        input_shape = self.base_model.input_shape[1:]
        input_values = [keras.Input(shape=input_shape) for _ in range(self.arity)]
        if self.arity == 2:
            # siamese structure
            head_layer = HatLayer()(*(self._base_model(v) for v in input_values))
            wrapped_model = Model(inputs=input_values, outputs=head_layer)
        elif self.arity == 3:
            # triplet structure
            raise NotImplementedError
        else:
            raise NotImplementedError

        wrapped_model.compile(loss=self.loss)

        return wrapped_model

    def fit(
            self, doc_array: Union[DocumentArray, DocumentArrayMemmap], **kwargs
    ) -> None:
        wrapped_model = self._compile()
        # wrapped_model.fit()
