from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from jina.helper import cached_property
from jina.logging.profile import ProgressBar
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from . import head_layers, datasets
from ..base import BaseTuner, BaseHead, BaseArityModel
from ...helper import DocumentArrayLike
from ..dataset.helper import get_dataset


class _ArityModel(BaseArityModel, nn.Module):
    ...


class PytorchTuner(BaseTuner):
    @property
    def head_layer(self) -> BaseHead:
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)
        elif isinstance(self._head_layer, BaseHead):
            return self._head_layer

    @cached_property
    def wrapped_model(self) -> nn.Module:
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

    def _eval(self, data, description: str = 'Evaluating'):
        self.wrapped_model.eval()

        losses = []
        metrics = []

        get_desc_str = (
            lambda: f'Loss={np.mean(losses):.2f} Accuracy={np.mean(metrics):.2f}'
        )

        with ProgressBar(description, message_on_done=get_desc_str) as p:
            for inputs, label in data:
                with torch.inference_mode():
                    outputs = self.wrapped_model(*inputs)
                    loss = self.wrapped_model.loss_fn(outputs, label)
                    metric = self.wrapped_model.metric_fn(outputs, label)

                losses.append(loss.item())
                metrics.append(metric.numpy())

                p.update(message=get_desc_str())

        return losses, metrics

    def _train(self, data, optimizer: Optimizer, description: str):

        self.wrapped_model.train()

        losses = []
        metrics = []

        get_desc_str = (
            lambda: f'Loss={np.mean(losses):.2f} Accuracy={np.mean(metrics):.2f}'
        )

        with ProgressBar(description, message_on_done=get_desc_str) as p:
            for inputs, label in data:
                # forward step
                outputs = self.wrapped_model(*inputs)
                loss = self.wrapped_model.loss_fn(outputs, label)
                metric = self.wrapped_model.metric_fn(outputs, label)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                metrics.append(metric.numpy())

                p.update(message=get_desc_str())
        return losses, metrics

    def fit(
        self,
        train_data: DocumentArrayLike,
        eval_data: Optional[DocumentArrayLike] = None,
        epochs: int = 10,
        batch_size: int = 256,
        **kwargs,
    ):
        optimizer = torch.optim.RMSprop(
            params=self.wrapped_model.parameters()
        )  # stay the same as keras

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

                le, me = self._eval(_data)
                losses_eval.extend(le)
                metrics_eval.extend(me)

        return {
            'loss': {'train': losses_train, 'eval': losses_eval},
            'metric': {'train': metrics_train, 'eval': metrics_eval},
        }

    def save(self, *args, **kwargs):
        torch.save(self.embed_model.state_dict(), *args, **kwargs)
