from typing import Optional

import torch
import torch.nn as nn
from jina.logging.profile import ProgressBar
from torch.utils.data.dataloader import DataLoader

from . import head_layers, datasets
from ..base import BaseTrainer, BaseHead, BaseArityModel, DocumentArrayLike
from ..helper import get_dataset


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

    def _get_data_loader(self, inputs, batch_size: int, shuffle: bool):
        ds = get_dataset(datasets, self.arity)
        return DataLoader(
            dataset=ds(inputs=inputs),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _eval(self, data, model: nn.Module, pbar_description: str):
        model.eval()

        losses = []
        metrics = []

        get_desc_str = (
            lambda: f'Loss={float(sum(losses) / len(losses)):.2f} Accuracy={float(sum(metrics) / len(metrics)):.2f}'
        )

        with ProgressBar(pbar_description, on_done=get_desc_str) as p:
            for inputs, label in data:
                outputs = model(*inputs)
                loss = model.loss_fn(outputs, label)
                metric = model.metric_fn(outputs, label)

                losses.append(loss.item())
                metrics.append(metric.numpy())

                p.update(details=get_desc_str())

    def _train(self, data, model: nn.Module, optimizer, pbar_description: str):

        model.train()

        losses = []
        metrics = []

        get_desc_str = (
            lambda: f'Loss={float(sum(losses) / len(losses)):.2f} Accuracy={float(sum(metrics) / len(metrics)):.2f}'
        )

        with ProgressBar(pbar_description, on_done=get_desc_str) as p:
            for inputs, label in data:
                # forward step
                outputs = model(*inputs)
                loss = model.loss_fn(outputs, label)
                metric = model.metric_fn(outputs, label)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                metrics.append(metric.numpy())

                p.update(details=get_desc_str())

    def fit(
        self,
        train_data: DocumentArrayLike,
        eval_data: Optional[DocumentArrayLike] = None,
        epochs: int = 10,
        batch_size: int = 256,
        **kwargs,
    ) -> None:
        model = self.wrapped_model

        optimizer = torch.optim.RMSprop(
            params=model.parameters()
        )  # stay the same as keras

        for epoch in range(epochs):
            _data = self._get_data_loader(
                inputs=train_data, batch_size=batch_size, shuffle=False
            )

            self._train(
                _data,
                model,
                optimizer,
                pbar_description=f'Epoch {epoch + 1}/{epochs}',
            )

            if eval_data:
                _data = self._get_data_loader(
                    inputs=eval_data, batch_size=batch_size, shuffle=False
                )

                self._eval(_data, model, pbar_description='Evaluating...')

    def save(self, *args, **kwargs):
        torch.save(self.base_model, *args, **kwargs)
