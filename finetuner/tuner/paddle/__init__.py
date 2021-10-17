from typing import Dict, Optional, Union

import paddle
from jina.logging.profile import ProgressBar
from paddle import nn
from paddle.io import DataLoader
from paddle.optimizer import Optimizer

from . import losses, datasets
from ..base import BaseTuner, BaseLoss
from ...helper import DocumentArrayLike
from ..dataset.helper import get_dataset
from ..logger import LogGenerator


class PaddleTuner(BaseTuner):
    def _get_loss(self, loss: Union[BaseLoss, str]):
        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, BaseLoss):
            return loss

    def _get_data_loader(self, inputs, batch_size: int, shuffle: bool):
        ds = get_dataset(datasets, self.arity)
        return DataLoader(
            dataset=ds(inputs=inputs),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _get_optimizer(
        self, optimizer: str, optimizer_kwargs: Optional[dict], learning_rate: float
    ) -> Optimizer:
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
    ):

        if device == 'cuda':
            paddle.set_device('gpu:0')
        elif device == 'cpu':
            paddle.set_device('cpu')
        else:
            raise ValueError(f'Device {device} not recognized')

        _optimizer = self._get_optimizer(optimizer, optimizer_kwargs, learning_rate)

        losses_train = []
        losses_eval = []

        for epoch in range(epochs):
            _data = self._get_data_loader(
                inputs=train_data, batch_size=batch_size, shuffle=False
            )
            lt = self._train(
                _data,
                _optimizer,
                description=f'Epoch {epoch + 1}/{epochs}',
            )
            losses_train.extend(lt)

            if eval_data:
                _data = self._get_data_loader(
                    inputs=eval_data, batch_size=batch_size, shuffle=False
                )

                le = self._eval(_data, train_log=LogGenerator('T', lt)())
                losses_eval.extend(le)

        return {'loss': {'train': losses_train, 'eval': losses_eval}}

    def save(self, *args, **kwargs):
        paddle.save(self.embed_model.state_dict(), *args, **kwargs)
