from copy import deepcopy
from typing import Union, Optional

import torch.nn as nn
from jina.logging.logger import JinaLogger

from ..base import BaseTrainer
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
        pass

    def fit(
        self,
        inputs,
        **kwargs,
    ) -> None:
        best_model_wts = deepcopy(self.base_model.state_dict())
        best_acc = 0.0
        num_epochs = self.params['num_epochs']

        for epoch in self.params['num_epochs']:
            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}')
            self.base_model.train()

            running_loss = 0.0
            running_corrects = 0

    def save(self, *args, **kwargs):
        pass

    @classmethod
    def load(self, path: str, *args, **kwargs):
        pass
