from typing import Union, Callable

import paddle
from jina.logging.profile import ProgressBar
from paddle import nn
from paddle.io import DataLoader
from paddle.io import IterableDataset

from . import head_layers
from ..base import BaseTrainer, DocumentArrayLike, BaseHead, BaseDataset


class _ArityModel(nn.Layer):
    """The helper class to copy the network for multi-inputs."""

    def __init__(self, base_model: nn.Layer):
        super().__init__()
        self._base_model = base_model

    def forward(self, *args):
        return tuple(self._base_model(a) for a in args)


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

        if self.arity == 2:

            from ..dataset import SiameseMixin

            class _SiameseDataset(SiameseMixin, BaseDataset, IterableDataset):
                ...

            ds = _SiameseDataset
        elif self.arity == 3:
            from ..dataset import TripletMixin, BaseDataset

            class _TripletDataset(TripletMixin, BaseDataset, IterableDataset):
                ...

            ds = _TripletDataset
        else:
            raise NotImplementedError

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

            data_loader = self._get_data_loader(inputs=train_data)
            with ProgressBar(task_name=f'Epoch {epoch + 1}/{epochs}') as p:
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

                    p.update()

                self.logger.info(
                    f'Training: Loss={sum(losses) / len(losses)} Accuracy={metric}'
                )

    def save(self, save_path: str, input_spec: Union[list, tuple] = None):
        base_model = paddle.jit.to_static(self.base_model, input_spec=input_spec)
        paddle.jit.save(base_model, save_path)
