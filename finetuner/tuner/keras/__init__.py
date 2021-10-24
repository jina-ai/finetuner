from typing import Dict, Optional, Union

import tensorflow as tf
from jina.logging.profile import ProgressBar
from tensorflow import keras
from tensorflow.keras.optimizers import Optimizer

from . import losses, datasets
from ..base import BaseTuner, BaseLoss
from ..dataset.helper import get_dataset
from ..summary import ScalarSequence, Summary
from ...helper import DocumentArrayLike, AnyDataLoader


class KerasTuner(BaseTuner):
    def _get_loss(self, loss: Union[BaseLoss, str]) -> BaseLoss:
        """Get the loss layer."""
        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, BaseLoss):
            return loss

    def _get_data_loader(
        self, inputs: DocumentArrayLike, batch_size: int, shuffle: bool
    ) -> AnyDataLoader:
        """Get tensorflow ``Dataset`` from the input data."""

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

    def _train(
        self, data: AnyDataLoader, optimizer: Optimizer, description: str
    ) -> ScalarSequence:
        """Train the model on given labeled data"""

        _summary = ScalarSequence('Train Loss')
        with ProgressBar(
            description,
            message_on_done=_summary.__str__,
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

                _summary += loss.numpy()

                p.update(message=str(_summary))
                self._train_data_len += 1

        return _summary

    def _eval(
        self,
        data: AnyDataLoader,
        description: str = 'Evaluating',
        train_loss: Optional[ScalarSequence] = None,
    ) -> ScalarSequence:
        """Evaluate the model on given labeled data"""

        _summary = ScalarSequence('Eval Loss')

        with ProgressBar(
            description,
            message_on_done=lambda: f'{train_loss} | {_summary}',
            total_length=self._eval_data_len,
        ) as p:
            self._eval_data_len = 0
            for inputs, label in data:
                embeddings = [self._embed_model(inpt) for inpt in inputs]
                loss = self._loss([*embeddings, label])

                _summary += loss.numpy()

                p.update(message=str(_summary))
                self._eval_data_len += 1

        return _summary

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
    ) -> Summary:
        """Finetune the model on the training data.

        :param train_data: Data on which to train the model
        :param eval_data: Data on which to evaluate the model at the end of each epoch
        :param epochs: Number of epochs to train the model
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

        _optimizer = self._get_optimizer(optimizer, optimizer_kwargs, learning_rate)

        m_train_loss = ScalarSequence('train')
        m_eval_loss = ScalarSequence('eval')

        with get_device(device):
            for epoch in range(epochs):
                lt = self._train(
                    _train_data,
                    _optimizer,
                    description=f'Epoch {epoch + 1}/{epochs}',
                )
                m_train_loss += lt

                if eval_data:
                    le = self._eval(_eval_data, train_loss=m_train_loss)
                    m_eval_loss += le

        return Summary(m_train_loss, m_eval_loss)

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
