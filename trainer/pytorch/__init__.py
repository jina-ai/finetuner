from typing import Union, Optional

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from jina.logging.logger import JinaLogger

from . import head_layers
from ..base import BaseTrainer
from .dataset import JinaSiameseDataset
from .networks import SiameseInputsLayer
from .head_layers import HeadLayer


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
        return self._loss or self.head_layer.recommended_loss

    @property
    def wrapped_model(self) -> nn.Module:
        if self.base_model is None:
            raise ValueError(f'base_model is not set')

        net = SiameseInputsLayer(base_model=self.base_model)  # build module
        net = self.head_layer(net)  # wrap with head layer
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
        criterion = self.head_layer.recommended_loss
        num_epochs = 10

        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')
            model.train()

            losses = []
            correct = 0
            total = 0

            for (l_input, r_input), label in data_loader:
                l_input, r_input, label = map(
                    lambda x: x.to(device), [l_input, r_input, label]
                )

                prob = model(l_input, r_input)
                loss = criterion(prob, label)
                eval_sign = torch.eq(torch.sign(prob), label)
                correct = torch.count_nonzero(eval_sign).item()
                total = float(len(eval_sign))

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
            self.logger.info(
                "Training: Loss={:.2f} Accuracy={:.2f}".format(
                    sum(losses) / len(losses), correct / total
                )
            )
        # TODO eval phrase

    def save(self, *args, **kwargs):
        torch.save(self.base_model.state_dict(), *args, **kwargs)
