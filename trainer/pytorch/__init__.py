import torch
import torch.nn as nn
from jina.logging.profile import ProgressBar
from torch.utils.data.dataloader import DataLoader

from . import head_layers
from .dataset import JinaSiameseDataset
from .head_layers import HeadLayer
from ..base import BaseTrainer


class _ArityModel(nn.Module):
    """The helper class to copy the network for multi-inputs. """

    def __init__(self, base_model):
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
        return DataLoader(
            dataset=JinaSiameseDataset(inputs=inputs),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def fit(
        self,
        doc_array,
        epochs: int,
        **kwargs,
    ) -> None:
        model = self.wrapped_model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        data_loader = self._get_data_loader(inputs=doc_array)

        optimizer = torch.optim.RMSprop(
            params=model.parameters()
        )  # stay the same as keras
        loss_fn = self.head_layer.default_loss

        for epoch in range(epochs):
            model.train()

            losses = []
            correct, total = 0, 0
            with ProgressBar(task_name=f'Epoch {epoch+1}/{epochs}') as p:
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
                    correct += torch.count_nonzero(eval_sign).item()
                    total += len(eval_sign)

                    p.update()

                self.logger.info(
                    "Training: Loss={:.2f} Accuracy={:.2f}".format(
                        sum(losses) / len(losses), correct / total
                    )
                )

    def save(self, *args, **kwargs):
        torch.save(self.base_model.state_dict(), *args, **kwargs)
