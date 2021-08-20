from typing import Union, Callable

import torch
import torch.nn as nn
from jina.logging.profile import ProgressBar
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader

from . import head_layers
from .head_layers import HeadLayer
from ..base import BaseTrainer, DocumentArrayLike


class _ArityModel(nn.Module):
    """The helper class to copy the network for multi-inputs."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self._base_model = base_model

    def forward(self, *args):
        return tuple(self._base_model(a) for a in args)


class PytorchTrainer(BaseTrainer):
    @property
    def head_layer(self) -> HeadLayer:
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)
        elif isinstance(self._head_layer, HeadLayer):
            return self._head_layer

    @property
    def wrapped_model(self) -> nn.Module:
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
        **kwargs,
    ) -> None:
        model = self.wrapped_model
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device)

        optimizer = torch.optim.RMSprop(
            params=model.parameters()
        )  # stay the same as keras

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

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())

                    p.update()

                self.logger.info(
                    f'Training: Loss={sum(losses) / len(losses)} Accuracy={metric}'
                )

    def save(self, *args, **kwargs):
        torch.save(self.base_model, *args, **kwargs)
