from typing import Union, Callable

import paddle
from jina.logging.profile import ProgressBar
from paddle import nn
from paddle.io import DataLoader

from . import head_layers, datasets
from ..helper import get_dataset
from ..base import BaseTrainer, DocumentArrayLike, BaseHead, BaseArityModel


class _ArityModel(BaseArityModel, nn.Layer):
    ...


class PaddleTrainer(BaseTrainer):
    @property
    def head_layer(self) -> BaseHead:
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)
        elif isinstance(self._head_layer, nn.Layer):
            return self._head_layer

    @property
    def wrapped_model(self) -> nn.Layer:
        if self.base_model is None:
            raise ValueError(f'base_model is not set')

        return self.head_layer(_ArityModel(self.base_model))  # wrap with head layer

    def _get_data_loader(self, inputs, batch_size=256, shuffle=False):
        ds = get_dataset(datasets, self.arity)
        return DataLoader(
            dataset=ds(inputs=inputs),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def fit(
        self,
        train_data: Union[
            DocumentArrayLike,
            Callable[..., DocumentArrayLike],
        ],
        epochs: int = 10,
        **kwargs,
    ):
        model = self.wrapped_model

        optimizer = paddle.optimizer.RMSProp(
            learning_rate=0.01, parameters=model.parameters()
        )

        for epoch in range(epochs):
            model.train()

            losses = []
            metrics = []

            data_loader = self._get_data_loader(inputs=train_data)
            with ProgressBar(f'Epoch {epoch + 1}/{epochs}') as p:
                for inputs, label in data_loader:
                    # forward step
                    outputs = model(*inputs)

                    loss = model.loss_fn(outputs[0], label)
                    metric = model.metric_fn(outputs[1], label)

                    # clean gradients
                    optimizer.clear_grad()

                    loss.backward()
                    optimizer.step()

                    losses.append(loss.numpy())
                    metrics.append(metric)

                    p.update()

                self.logger.info(
                    f'Training: Loss={sum(losses) / len(losses)} Accuracy={sum(metrics) / len(metrics)}'
                )

    def save(self, save_path: str, input_spec: Union[list, tuple] = None):
        base_model = paddle.jit.to_static(self.base_model, input_spec=input_spec)
        paddle.jit.save(base_model, save_path)
