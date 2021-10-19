from typing import Dict, Optional, Union

import numpy as np
import paddle
from jina.logging.profile import ProgressBar
from paddle.io import DataLoader
from paddle.optimizer import Optimizer

from . import losses, datasets
from ..base import BaseTuner, BaseLoss
from ...helper import DocumentArrayLike
from ..dataset.helper import get_dataset
from ..logger import LogGenerator
from ..stats import TunerStats


class PaddleTuner(BaseTuner):
    def _get_loss(self, loss: Union[BaseLoss, str]):
        """Get the loss layer."""

        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, BaseLoss):
            return loss

    def _get_data_loader(self, inputs, batch_size: int, shuffle: bool):
        """Get the paddle ``DataLoader`` from the input data. """

        ds = get_dataset(datasets, self.arity)
        return DataLoader(
            dataset=ds(inputs=inputs, catalog=self._catalog),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _get_optimizer(
        self, optimizer: str, optimizer_kwargs: Optional[dict], learning_rate: float
    ) -> Optimizer:
        """Get the optimizer for training."""

        params = self._embed_model.parameters()
        optimizer_kwargs = self._get_optimizer_kwargs(optimizer, optimizer_kwargs)

        if optimizer == 'adam':
            return paddle.optimizer.Adam(
                parameters=params,
                learning_rate=learning_rate,
                beta1=optimizer_kwargs['beta_1'],
                beta2=optimizer_kwargs['beta_2'],
                epsilon=optimizer_kwargs['epsilon'],
            )
        elif optimizer == 'rmsprop':
            return paddle.optimizer.RMSProp(
                parameters=params, learning_rate=learning_rate, **optimizer_kwargs
            )
        elif optimizer == 'sgd':
            return paddle.optimizer.Momentum(
                parameters=params,
                learning_rate=learning_rate,
                momentum=optimizer_kwargs['momentum'],
                use_nesterov=optimizer_kwargs['nesterov'],
            )

    def _eval(self, data, description: str = 'Evaluating', train_log: str = ''):
        """Evaluate the model on given labeled data"""

        self._embed_model.eval()

        losses = []

        log_generator = LogGenerator('E', losses, train_log)

        with ProgressBar(
            description, message_on_done=log_generator, total_length=self._eval_data_len
        ) as p:
            self._eval_data_len = 0
            for inputs, label in data:
                embeddings = [self._embed_model(inpt) for inpt in inputs]
                loss = self._loss(embeddings, label)

                losses.append(loss.item())

                p.update(message=log_generator())
                self._eval_data_len += 1

        return losses

    def _train(self, data, optimizer: Optimizer, description: str):
        """Train the model on given labeled data"""

        self._embed_model.train()

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
                # forward step
                embeddings = [self._embed_model(inpt) for inpt in inputs]
                loss = self._loss(embeddings, label)

                optimizer.clear_grad()

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                p.update(message=log_generator())
                self._train_data_len += 1
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

        if device == 'cuda':
            paddle.set_device('gpu:0')
        elif device == 'cpu':
            paddle.set_device('cpu')
        else:
            raise ValueError(f'Device {device} not recognized')

        _optimizer = self._get_optimizer(optimizer, optimizer_kwargs, learning_rate)

        stats = TunerStats()

        for epoch in range(epochs):
            _data = self._get_data_loader(
                inputs=train_data, batch_size=batch_size, shuffle=False
            )
            lt = self._train(
                _data,
                _optimizer,
                description=f'Epoch {epoch + 1}/{epochs}',
            )
            stats.add_train_loss(lt)

            if eval_data:
                _data = self._get_data_loader(
                    inputs=eval_data, batch_size=batch_size, shuffle=False
                )

                le = self._eval(_data, train_log=LogGenerator('T', lt)())
                stats.add_eval_loss(le)
                stats.add_eval_metric(self.get_metrics(eval_data))

            stats.print_last()
        return stats

    def get_embeddings(self, data: DocumentArrayLike):
        blobs = data.blobs
        embeddings = self.embed_model(paddle.Tensor(blobs))
        for doc, embed in zip(data, embeddings):
            doc.embedding = np.array(embed)

    def save(self, *args, **kwargs):
        """Save the embedding model.

        You need to pass the path where to save the model in either ``args`` or
        ``kwargs`` (for ``path`` key).

        :param args: Arguments to pass to ``paddle.save`` function
        :param kwargs: Keyword arguments to pass to ``paddle.save`` function
        """
        paddle.save(self.embed_model.state_dict(), *args, **kwargs)
