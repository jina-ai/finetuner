from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Union

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import DataLoader

from ... import __default_tag_key__
from ..base import BaseTuner
from ..state import TunerState
from . import losses
from .datasets import PytorchClassDataset, PytorchInstanceDataset, PytorchSessionDataset

if TYPE_CHECKING:
    from docarray import DocumentArray

    from ...helper import CollateFnType, PreprocFnType


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


class CollateAll:
    def __init__(self, content_collate_fn: 'CollateFnType'):
        self._content_collate_fn = content_collate_fn

    def __call__(self, inputs):
        batch_content = self._content_collate_fn([x[0] for x in inputs])
        batch_labels = default_collate([x[1] for x in inputs])
        return batch_content, batch_labels


class PytorchTuner(BaseTuner[nn.Module, DataLoader, Optimizer, _LRScheduler]):
    def _get_loss(self, loss: Union[nn.Module, str]) -> nn.Module:
        """Get the loss layer."""
        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, nn.Module):
            return loss

    def _get_data_loader(
        self,
        data: 'DocumentArray',
        batch_size: int,
        shuffle: bool,
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
        num_items_per_class: Optional[int] = None,
        num_workers: int = 0,
    ) -> DataLoader:
        """Get the dataloader for the dataset."""

        if collate_fn:
            collate_fn_all = CollateAll(collate_fn)
        else:
            collate_fn_all = None

        if __default_tag_key__ in data[0].tags:
            dataset = PytorchClassDataset(data, preprocess_fn=preprocess_fn)
        else:
            if len(data[0].matches) > 0:
                dataset = PytorchSessionDataset(data, preprocess_fn=preprocess_fn)
            else:
                dataset = PytorchInstanceDataset(data, preprocess_fn=preprocess_fn)

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
        """Move the model to device and set device."""
        self.device = get_device(self._device_name)
        self._embed_model = self._embed_model.to(self.device)

    def _default_configure_optimizer(self, model: nn.Module) -> Optimizer:
        """Get the default Adam optimizer."""
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)
        return optimizer

    def _train(self, data: DataLoader):
        """Train the model on the given labeled data."""

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

    def _eval(self, data: DataLoader):
        """Compute the validation loss on the given labeled data."""

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

    def _fit(
        self,
        train_data: 'DocumentArray',
        eval_data: Optional['DocumentArray'] = None,
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
        epochs: int = 10,
        batch_size: int = 256,
        num_items_per_class: Optional[int] = None,
        num_workers: int = 0,
    ):
        """Fit the model - training and evaluation."""

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

    def save(self, *args, **kwargs):
        """Save the embedding model.

        You need to pass the path where to save the model in either ``args`` or
        ``kwargs`` (for ``f`` key).

        :param args: Arguments to pass to ``torch.save`` function.
        :param kwargs: Keyword arguments to pass to ``torch.save`` function.
        """
        torch.save(self.embed_model.state_dict(), *args, **kwargs)


def get_device(device: str):
    """Get Pytorch compute device.

    :param device: device name.
    """
    # translate our own alias into framework-compatible ones
    return torch.device(device)
