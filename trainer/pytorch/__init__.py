from copy import deepcopy
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from jina.logging.logger import JinaLogger

from ..base import BaseTrainer
from .networks import SiameseNet
from .dataset import JinaSiameseDataset


class PytorchTrainer(BaseTrainer):
    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        arity: int = 2,
        head_model: Union[nn.Module, str, None] = 'HatLayer',
        loss: str = 'hinge',
        params: dict = {
            'criterion': 1,
            'optimizer': 2,
            'scheduler': 3,
            'num_epochs': 20,
            'batch_size': 4,
            'shuffle': True,
            'num_workers': 1,
            'device': 'cpu',
        },
        **kwargs,
    ):
        super().__init__(base_model, arity, head_model, loss, **kwargs)
        self.params = params
        self.logger = JinaLogger(self.__class__.__name__)

    @property
    def base_model(self) -> nn.Module:
        return self._base_model

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def head_model(self) -> nn.Module:
        return self._head_model

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def wrapped_model(self) -> nn.Module:
        if self.arity == 2:
            return SiameseNet(base_model=self.base_model)
        else:
            raise NotImplementedError

    def _get_data_loader(self, inputs):
        return DataLoader(
            dataset=JinaSiameseDataset(inputs=inputs),
            batch_size=self.params['batch_size'],
            shuffle=self.params['shuffle'],
            num_workers=self.params['num_workers'],
        )

    def fit(
        self,
        inputs,
        phase='train',
        **kwargs,
    ) -> None:
        best_model_wts = deepcopy(self.wrapped_model.state_dict())
        best_acc = 0.0
        data_loader = self._get_data_loader(inputs=inputs)
        optimizer = self.params['optimizer']
        criterion = self.params['criterion']
        num_epochs = self.params['num_epochs']
        scheduler = self.params['scheduler']

        for epoch in range(num_epochs):
            self.logger(f'Epoch {epoch}/{num_epochs - 1}')
            self.wrapped_model.train()

            running_loss = 0.0
            running_corrects = 0

            for pairs, labels in data_loader:
                pairs.to(self.device)
                labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.wrapped_model(*pairs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            scheduler.step()

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            self.logger.info('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(self.wrapped_model.state_dict())
        self.logger.info(f'Best val Acc: {best_acc}')
        self.wrapped_model.load_state_dict(best_model_wts)
        return self.wrapped_model

    def save(self, *args, **kwargs):
        pass

    @classmethod
    def load(self, path: str, *args, **kwargs):
        pass
