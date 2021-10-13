from typing import Optional

import paddle
from jina.logging.profile import ProgressBar
from paddle import nn
from paddle.io import DataLoader
from paddle.optimizer import Optimizer
from . import head_layers, datasets
from ..base import BaseTuner, BaseHead, BaseArityModel
from ...helper import DocumentArrayLike
from ..dataset.helper import get_dataset
from ..logger import LogGenerator


class _ArityModel(BaseArityModel, nn.Layer):
    ...


class PaddleTuner(BaseTuner):
    @property
    def head_layer(self) -> BaseHead:
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)
        elif isinstance(self._head_layer, nn.Layer):
            return self._head_layer

    @property
    def wrapped_model(self) -> nn.Layer:
        if self.embed_model is None:
            raise ValueError(f'embed_model is not set')

        return self.head_layer(_ArityModel(self.embed_model))  # wrap with head layer

    def _get_data_loader(self, inputs, batch_size: int, shuffle: bool):
        ds = get_dataset(datasets, self.arity)
        return DataLoader(
            dataset=ds(inputs=inputs),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _eval(self, data, description: str = 'Evaluating', train_log: str = ''):
        self.wrapped_model.eval()

        losses = []
        metrics = []

        log_generator = LogGenerator('E', losses, metrics, train_log)

        with ProgressBar(description, message_on_done=log_generator) as p:
            for inputs, label in data:
                outputs = self.wrapped_model(*inputs)
                loss = self.wrapped_model.loss_fn(outputs, label)
                metric = self.wrapped_model.metric_fn(outputs, label)

                losses.append(loss.item())
                metrics.append(metric.numpy())

                p.update(message=log_generator())

        return losses, metrics

    def _train(self, data, optimizer: Optimizer, description: str):

        self.wrapped_model.train()

        losses = []
        metrics = []

        log_generator = LogGenerator('T', losses, metrics)

        with ProgressBar(
            description, message_on_done=log_generator, final_line_feed=False
        ) as p:
            for inputs, label in data:
                # forward step
                outputs = self.wrapped_model(*inputs)
                loss = self.wrapped_model.loss_fn(outputs, label)
                metric = self.wrapped_model.metric_fn(outputs, label)

                optimizer.clear_grad()

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                metrics.append(metric.numpy())

                p.update(message=log_generator())
        return losses, metrics

    def fit(
        self,
        train_data: DocumentArrayLike,
        eval_data: Optional[DocumentArrayLike] = None,
        epochs: int = 10,
        batch_size: int = 256,
        device: str = 'cpu',
        **kwargs,
    ):
        optimizer = paddle.optimizer.RMSProp(
            learning_rate=0.01, parameters=self.wrapped_model.parameters()
        )

        if device == 'cuda':
            paddle.set_device('gpu:0')
        elif device == 'cpu':
            paddle.set_device('cpu')
        else:
            raise ValueError(f'Device {device} not recognized')

        losses_train = []
        metrics_train = []
        losses_eval = []
        metrics_eval = []

        for epoch in range(epochs):
            _data = self._get_data_loader(
                inputs=train_data, batch_size=batch_size, shuffle=False
            )
            lt, mt = self._train(
                _data,
                optimizer,
                description=f'Epoch {epoch + 1}/{epochs}',
            )
            losses_train.extend(lt)
            metrics_train.extend(mt)

            if eval_data:
                _data = self._get_data_loader(
                    inputs=eval_data, batch_size=batch_size, shuffle=False
                )

                le, me = self._eval(_data, train_log=LogGenerator('T', lt, mt)())
                losses_eval.extend(le)
                metrics_eval.extend(me)

        return {
            'loss': {'train': losses_train, 'eval': losses_eval},
            'metric': {'train': metrics_train, 'eval': metrics_eval},
        }

    def save(self, *args, **kwargs):
        paddle.save(self.embed_model.state_dict(), *args, **kwargs)
