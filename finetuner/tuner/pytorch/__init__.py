from typing import (TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence,
                    Union)

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import DataLoader

from ... import __default_tag_key__
from ...excepts import DimensionMismatchException
from ..base import BaseTuner
from ..state import TunerState
from . import losses
from .datasets import PytorchClassDataset, PytorchSessionDataset

if TYPE_CHECKING:
    from ...helper import CollateFnType, DocumentSequence, PreprocFnType


def _to_device(
    inputs: Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]],
    device: torch.device,
) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    elif isinstance(inputs, Mapping):
        return {k: v.to(device) for k, v in inputs.items()}
    elif isinstance(inputs, Sequence):
        return [x.to(device) for x in inputs]


class PytorchTuner(BaseTuner[nn.Module, DataLoader, Optimizer, _LRScheduler]):
    def _get_loss(self, loss: Union[nn.Module, str]) -> nn.Module:
        """Get the loss layer."""
        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, nn.Module):
            return loss

    def _get_data_loader(
        self,
        data: 'DocumentSequence',
        batch_size: int,
        shuffle: bool,
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
        num_items_per_class: Optional[int] = None,
        num_workers: int = 0,
    ) -> DataLoader:
        """Get the dataloader for the dataset"""

        if collate_fn:

            def collate_fn_all(inputs):
                batch_content = collate_fn([x[0] for x in inputs])
                batch_labels = default_collate([x[1] for x in inputs])
                return batch_content, batch_labels

        else:
            collate_fn_all = None

        if __default_tag_key__ in data[0].tags:
            dataset = PytorchClassDataset(data, preprocess_fn=preprocess_fn)
        else:
            dataset = PytorchSessionDataset(data, preprocess_fn=preprocess_fn)

        batch_sampler = self._get_batch_sampler(
            dataset,
            batch_size,
            shuffle=shuffle,
            num_items_per_class=num_items_per_class,
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn_all,
            num_workers=num_workers,
        )

        return data_loader

    def _move_model_to_device(self):
        """Move the model to device and set device"""
        self.device = get_device(self._device_name)
        self._embed_model = self._embed_model.to(self.device)

    def _attach_projection_head(
        self,
        output_dim: Optional[int] = 128,
        num_layers: Optional[int] = 3,
        num_channels: Optional[int] = 3,
    ):
        """Attach a projection head on top of the embed model for self-supervised learning.
        Calling this function will modify :attr:`self._embed_model` in place.

        :param output_dim: The output dimensionality of the projection, default 128, recommend 32, 64, 128, 256.
        :param num_layers: Number of layers of the projection head, default 3, recommend 2, 3.
        :param num_channels: Generate a fake image to interpret the output dim of embed model. Grey scale image should be 1.
        """
        # interpret embed model output shape
        output = self._embed_model(torch.rand(1, num_channels, 224, 224))
        if not len(output.size()) == 2:
            raise DimensionMismatchException(
                f'Expected input shape is 2d, got {len(output.size())}.'
            )
        projection_head = _ProjectionHead(output.shape[1], output_dim, num_layers)
        self._embed_model.add_module(
            projection_head
        )  # TODO add integration test when dataset ready.

    def _default_configure_optimizer(self, model: nn.Module) -> Optimizer:
        """Get the default Adam optimizer"""
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)
        return optimizer

    def _eval(self, data: DataLoader):
        """Evaluate the model on given labeled data"""

        self._embed_model.eval()

        for idx, (inputs, labels) in enumerate(data):
            self.state.batch_index = idx
            self._trigger_callbacks('on_val_batch_begin')

            inputs = _to_device(inputs, self.device)
            labels = _to_device(labels, self.device)

            with torch.no_grad():
                embeddings = self.embed_model(inputs)
                loss = self._loss(embeddings, labels)

            self.state.current_loss = loss.item()
            self._trigger_callbacks('on_val_batch_end')

    def _train(self, data: DataLoader):
        """Train the model on given labeled data"""

        self._embed_model.train()

        for idx, (inputs, labels) in enumerate(data):

            # Set state variables
            self.state.batch_index = idx
            for param_idx, param_group in enumerate(self._optimizer.param_groups):
                self.state.learning_rates[f'group_{param_idx}'] = param_group['lr']

            self._trigger_callbacks('on_train_batch_begin')
            inputs = _to_device(inputs, self.device)
            labels = _to_device(labels, self.device)

            embeddings = self.embed_model(inputs)
            loss = self._loss(embeddings, labels)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if self._scheduler_step == 'batch' and self._scheduler is not None:
                self._scheduler.step()

            self.state.current_loss = loss.item()
            self._trigger_callbacks('on_train_batch_end')

    def _fit(
        self,
        train_data: 'DocumentSequence',
        eval_data: Optional['DocumentSequence'] = None,
        epochs: int = 10,
        batch_size: int = 256,
        num_items_per_class: Optional[int] = None,
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
        num_workers: int = 0,
    ):
        # Get dataloaders
        train_dl = self._get_data_loader(
            train_data,
            batch_size=batch_size,
            num_items_per_class=num_items_per_class,
            shuffle=True,
            preprocess_fn=preprocess_fn,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        if eval_data:
            eval_dl = self._get_data_loader(
                eval_data,
                batch_size=batch_size,
                num_items_per_class=num_items_per_class,
                shuffle=False,
                preprocess_fn=preprocess_fn,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )

        # If self-supervised, add projection head, vision task.
        # TODO: change this when we merge dataset PR
        if isinstance(train_dl, 'InstanceDataset') and isinstance(
            eval_dl, 'InstanceDataset'
        ):
            # TODO interpret channels
            self._attach_projection_head(output_dim=128, num_layers=3, num_channels=3)
        # Set state
        self.state = TunerState(num_epochs=epochs)
        self._trigger_callbacks('on_fit_begin')

        for epoch in range(epochs):

            # Setting here as re-shuffling can change number of batches
            self.state.epoch = epoch
            self.state.num_batches_train = len(train_dl)
            self.state.batch_index = 0

            self._trigger_callbacks('on_epoch_begin')

            self._trigger_callbacks('on_train_epoch_begin')
            self._train(train_dl)

            if self._scheduler_step == 'epoch' and self._scheduler is not None:
                self._scheduler.step()

            self._trigger_callbacks('on_train_epoch_end')

            if eval_data:
                self.state.num_batches_val = len(eval_dl)
                self.state.batch_index = 0

                self._trigger_callbacks('on_val_begin')
                self._eval(eval_dl)
                self._trigger_callbacks('on_val_end')

            self._trigger_callbacks('on_epoch_end')

            if self.stop_training:
                break

        self._trigger_callbacks('on_fit_end')

        # If self-supervised, drop projection head, vision task.
        # TODO: change this when we merge dataset PR
        if isinstance(train_dl, 'InstanceDataset') and isinstance(
            eval_dl, 'InstanceDataset'
        ):
            del self._embed_model.projection_head

    def save(self, *args, **kwargs):
        """Save the embedding model.

        You need to pass the path where to save the model in either ``args`` or
        ``kwargs`` (for ``f`` key).

        :param args: Arguments to pass to ``torch.save`` function
        :param kwargs: Keyword arguments to pass to ``torch.save`` function
        """

        torch.save(self.embed_model.state_dict(), *args, **kwargs)


def get_device(device: str):
    """Get Pytorch compute device.

    :param device: device name
    """

    # translate our own alias into framework-compatible ones
    return torch.device(device)


class _ProjectionHead(nn.Module):
    """Projection head used internally for self-supervised training.
    It is (by default) a simple 3-layer MLP to be attached on top of embedding model only for training purpose.
    After training, it should be cut-out from the embedding model.
    """

    EPSILON = 1e-5

    def __init__(self, in_features: int, output_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.head_layers = nn.ModuleList()
        is_last_layer = False
        for idx in range(num_layers):
            if idx == num_layers - 1:
                is_last_layer = True
            if not is_last_layer:
                self.head_layers.append(
                    nn.Linear(
                        in_features=in_features, out_features=in_features, bias=False
                    )
                )
                self.head_layers.append(
                    nn.BatchNorm1d(num_features=in_features, eps=self.EPSILON)
                )
                self.head_layers.append(nn.ReLU())
            else:
                self.head_layers.append(
                    nn.Linear(
                        in_features=in_features, out_features=output_dim, bias=False
                    )
                )

    def forward(self, x):
        for layer in self.head_layers:
            x = layer(x)
        return x
