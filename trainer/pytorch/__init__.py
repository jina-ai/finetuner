from typing import Union, Optional

import torch
import torch.nn as nn
from jina.logging.logger import JinaLogger
from torch.utils.data.dataloader import DataLoader

from . import head_layers
from .dataset import JinaSiameseDataset
from .head_layers import HeadLayer
from ..base import BaseTrainer


class ArityModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self._base_model = base_model

    def forward(self, *args):
        return tuple(self._base_model(a) for a in args)


class PytorchTrainer(BaseTrainer):
    def __init__(
            self,
            base_model: Optional[nn.Module] = None,
            arity: int = 2,
            head_layer: Union[HeadLayer, str, None] = 'HatLayer',
            loss: str = 'hinge',
            **kwargs,
    ):
        super().__init__(base_model, arity, head_layer, loss, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)

    @property
    def base_model(self) -> nn.Module:
        return self._base_model

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def head_layer(self) -> HeadLayer:
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)
        elif isinstance(self._head_layer, HeadLayer):
            return self._head_layer

    @property
    def loss(self) -> str:
        return self._loss or self.head_layer.default_loss

    @property
    def wrapped_model(self) -> nn.Module:
        if self.base_model is None:
            raise ValueError(f'base_model is not set')

        net = self.head_layer(ArityModel(self.base_model))  # wrap with head layer
        return net

    def _get_data_loader(self, inputs, batch_size=256, shuffle=False, num_workers=1):
        return DataLoader(
            dataset=JinaSiameseDataset(inputs=inputs),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def fit(
            self,
            inputs,
            **kwargs,
    ) -> None:
        model = self.wrapped_model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        data_loader = self._get_data_loader(inputs=inputs)

        optimizer = torch.optim.RMSprop(
            params=model.parameters()
        )  # stay the same as keras
        loss_fn = self.head_layer.default_loss
        num_epochs = 10

        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')
            model.train()

            losses = []

            for (l_input, r_input), label in data_loader:
                l_input, r_input, label = map(
                    lambda x: x.to(device), [l_input, r_input, label]
                )

                head_value = model(l_input, r_input)
                loss = loss_fn(head_value, label)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            eval_sign = torch.eq(torch.sign(head_value), label)
            correct = torch.count_nonzero(eval_sign).item()
            total = len(eval_sign)

            self.logger.info(
                "Training: Loss={:.2f} Accuracy={:.2f}".format(
                    sum(losses) / len(losses), correct / total
                )
            )

    def save(self, *args, **kwargs):
        torch.save(self.base_model.state_dict(), *args, **kwargs)
