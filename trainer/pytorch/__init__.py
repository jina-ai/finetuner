from typing import Union, Callable

import torch
import torch.nn as nn
from jina.logging.profile import ProgressBar
from torch.utils.data.dataloader import DataLoader

from . import head_layers, datasets
from ..helper import get_dataset
from ..base import BaseTrainer, DocumentArrayLike, BaseHead, BaseArityModel


class _ArityModel(BaseArityModel, nn.Module):
    ...


class PytorchTrainer(BaseTrainer):
    @property
    def head_layer(self) -> BaseHead:
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)
        elif isinstance(self._head_layer, BaseHead):
            return self._head_layer

    @property
    def wrapped_model(self) -> nn.Module:
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
            metrics = []

            data_loader = self._get_data_loader(inputs=train_data)
            with ProgressBar(f'Epoch {epoch + 1}/{epochs}') as p:
                for inputs, label in data_loader:
                    # forward step
                    outputs = model(*inputs)
                    loss = model.loss_fn(outputs, label)
                    metric = model.metric_fn(outputs, label)

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())
                    metrics.append(metric.numpy())

                    p.update(
                        details=f'Loss={float(sum(losses) / len(losses)):.2f} Accuracy={float(sum(metrics) / len(metrics)):.2f}'
                    )

    def save(self, *args, **kwargs):
        torch.save(self.base_model, *args, **kwargs)
