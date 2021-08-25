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
        use_gpu: bool = False,
        **kwargs,
    ):
        if use_gpu:
            paddle.set_device('gpu')

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
                for i, (inputs, label) in enumerate(data_loader):
                    # forward step
                    outputs = model(*inputs)

                    loss = model.loss_fn(outputs, label)
                    metric = model.metric_fn(outputs, label)

                    # clean gradients
                    optimizer.clear_grad()

                    loss.backward()
                    optimizer.step()

                    losses.append(loss.numpy())
                    metrics.append(metric.numpy())

                    # p.update(
                    #     details=f'Loss={float(sum(losses) / len(losses)):.2f} Accuracy={float(sum(metrics) / len(metrics)):.2f}'
                    # )
                    p.update()
                    if i % 100 == 0:
                        print(
                            f'=> [Epoch {epoch+1}] Step={i} Loss={float(sum(losses) / len(losses)):.2f} Accuracy={float(sum(metrics) / len(metrics)):.2f}'
                        )

    def save(self, save_path: str, input_spec: Union[list, tuple] = None):
        base_static_model = None
        if hasattr(self.base_model, 'to_static'):
            base_static_model = self.base_model.to_static()

        if not base_static_model:
            base_static_model = paddle.jit.to_static(
                self.base_model, input_spec=input_spec
            )

        paddle.jit.save(base_static_model, save_path)
