from typing import TYPE_CHECKING, Optional, Union

import paddle
from paddle import nn
from paddle.fluid.dataloader.dataloader_iter import default_collate_fn
from paddle.io import DataLoader
from paddle.optimizer import Adam, Optimizer
from paddle.optimizer.lr import LRScheduler

from ... import __default_tag_key__
from ...device import get_device_paddle, to_device_paddle
from ..base import BaseTuner
from ..state import TunerState
from . import losses
from .datasets import PaddleClassDataset, PaddleInstanceDataset, PaddleSessionDataset

if TYPE_CHECKING:
    from docarray import DocumentArray

    from ...helper import CollateFnType, PreprocFnType


class PaddleTuner(BaseTuner[nn.Layer, DataLoader, Optimizer, LRScheduler]):
    def _get_loss(self, loss: Union[nn.Layer, str]) -> nn.Layer:
        """Get the loss layer."""
        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, nn.Layer):
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

            def collate_fn_all(inputs):
                batch_content = collate_fn([x[0] for x in inputs])
                batch_labels = default_collate_fn([x[1] for x in inputs])
                return batch_content, batch_labels

        else:
            collate_fn_all = None

        if __default_tag_key__ in data[0].tags:
            dataset = PaddleClassDataset(data, preprocess_fn=preprocess_fn)
        else:
            if len(data[0].matches) > 0:
                dataset = PaddleSessionDataset(data, preprocess_fn=preprocess_fn)
            else:
                dataset = PaddleInstanceDataset(data, preprocess_fn=preprocess_fn)

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
        self.device = get_device_paddle(self._device_name)
        self._embed_model.to(device=self.device)

    def _default_configure_optimizer(self, model: nn.Layer) -> Optimizer:
        """Get the default optimizer (Adam), if none was provided by user."""
        return Adam(parameters=model.parameters(), learning_rate=self._learning_rate)

    def _train(self, data: DataLoader):
        """Train the model on the given labeled data."""

        self._embed_model.train()

        for idx, (inputs, labels) in enumerate(data):

            # Set state variables
            self.state.learning_rates['learning_rate'] = self._optimizer.get_lr()
            self.state.batch_index = idx

            self._trigger_callbacks('on_train_batch_begin')

            inputs = to_device_paddle(inputs, self.device)
            labels = to_device_paddle(labels, self.device)

            embeddings = self.embed_model(inputs)
            loss = self._loss(embeddings, labels)

            self._optimizer.clear_grad()
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

            inputs = to_device_paddle(inputs, self.device)
            labels = to_device_paddle(labels, self.device)

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
        limit: int = 20,
        distance: str = 'cosine',
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
        ``kwargs`` (for ``path`` key).

        :param args: Arguments to pass to ``paddle.save`` function.
        :param kwargs: Keyword arguments to pass to ``paddle.save`` function.
        """
        paddle.save(self.embed_model.state_dict(), *args, **kwargs)
