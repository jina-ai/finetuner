from typing import Union, Callable

import paddle
from jina.logging.profile import ProgressBar
from paddle import nn
from paddle.io import DataLoader
from paddle.io import IterableDataset

from . import head_layers
from .head_layers import HeadLayer
from ..base import BaseTrainer, DocumentArrayLike


class _ArityModel(nn.Layer):
    """The helper class to copy the network for multi-inputs."""

    def __init__(self, base_model: nn.Layer):
        super().__init__()
        self._base_model = base_model

    def forward(self, *args):
        return tuple(self._base_model(a) for a in args)


class PaddleTrainer(BaseTrainer):
    @property
    def head_layer(self) -> HeadLayer:
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

            from ..dataset import SiameseMixin, Dataset

            class _SiameseDataset(SiameseMixin, Dataset, IterableDataset):
                ...

            ds = _SiameseDataset
        elif self.arity == 3:
            from ..dataset import TripletMixin, Dataset

            class _TripletDataset(TripletMixin, Dataset, IterableDataset):
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
        use_gpu: bool = False,
        **kwargs,
    ):
        model = self.wrapped_model
        if use_gpu:
            paddle.set_device('gpu')

        optimizer = paddle.optimizer.RMSProp(
            learning_rate=0.01, parameters=model.parameters()
        )

        loss_fn = self.head_layer.default_loss

        for epoch in range(epochs):
            model.train()

            losses = []
            correct, total = 0, 0

            data_loader = self._get_data_loader(inputs=train_data)
            with ProgressBar(task_name=f'Epoch {epoch + 1}/{epochs}') as p:
                for inputs, label in data_loader:
                    # forward step
                    head_value = model(*inputs)

                    loss = loss_fn(head_value, label)

                    # clean gradients
                    optimizer.clear_grad()

                    loss.backward()
                    optimizer.step()

                    losses.append(loss.numpy())

                    eval_sign = paddle.equal(
                        paddle.sign(head_value), paddle.sign(label)
                    )
                    correct += paddle.sum(paddle.nonzero(eval_sign)).numpy()
                    total += len(eval_sign)

                    p.update()

                self.logger.info(
                    f'Training: Loss={sum(losses) / len(losses)} Accuracy={float(correct / total)}'
                )

    def save(self, save_path: str, input_spec: Union[list, tuple] = None):
        base_model = paddle.jit.to_static(self.base_model, input_spec=input_spec)
        paddle.jit.save(base_model, save_path)
