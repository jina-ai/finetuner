from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Union

import paddle
from paddle import nn
from paddle.fluid.dataloader.dataloader_iter import default_collate_fn
from paddle.io import DataLoader
from paddle.optimizer import Adam, Optimizer
from paddle.optimizer.lr import LRScheduler

from ... import __default_tag_key__
from ..base import BaseTuner
from ..state import TunerState
from . import losses
from .datasets import PaddleClassDataset, PaddleSessionDataset

if TYPE_CHECKING:
    from ...helper import CollateFnType, DocumentSequence, PreprocFnType


def _to_device(
    inputs: Union[paddle.Tensor, Mapping[str, paddle.Tensor], Sequence[paddle.Tensor]],
    device,
) -> Union[paddle.Tensor, Dict[str, paddle.Tensor], List[paddle.Tensor]]:
    if isinstance(inputs, paddle.Tensor):
        return paddle.to_tensor(inputs, place=device)
    elif isinstance(inputs, Mapping):
        return {k: paddle.to_tensor(v, place=device) for k, v in inputs.items()}
    elif isinstance(inputs, Sequence):
        return [paddle.to_tensor(x, place=device) for x in inputs]


class PaddleTuner(BaseTuner[nn.Layer, DataLoader, Optimizer, LRScheduler]):
    def _get_loss(self, loss: Union[nn.Layer, str]) -> nn.Layer:
        """Get the loss layer."""
        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, nn.Layer):
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
                batch_labels = default_collate_fn([x[1] for x in inputs])
                return batch_content, batch_labels

        else:
            collate_fn_all = None

        if __default_tag_key__ in data[0].tags:
            dataset = PaddleClassDataset(data, preprocess_fn=preprocess_fn)
        else:
            dataset = PaddleSessionDataset(data, preprocess_fn=preprocess_fn)

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
        self._embed_model.to(device=self.device)

    def _default_configure_optimizer(self, model: nn.Layer) -> Optimizer:
        """Get the default optimizer (Adam), if none was provided by user."""

        return Adam(parameters=model.parameters(), learning_rate=self._learning_rate)

    def _eval(self, data: DataLoader):
        """Evaluate the model on given labeled data"""

        self._embed_model.eval()

        for idx, (inputs, labels) in enumerate(data):
            self.state.batch_index = idx
            self._trigger_callbacks('on_val_batch_begin')

            inputs = _to_device(inputs, self.device)
            labels = _to_device(labels, self.device)

            embeddings = self.embed_model(inputs)
            loss = self._loss(embeddings, labels)

            self.state.current_loss = loss.item()
            self._trigger_callbacks('on_val_batch_end')

    def _train(self, data: DataLoader):
        """Train the model on given labeled data"""

        self._embed_model.train()

        for idx, (inputs, labels) in enumerate(data):

            # Set state variables
            self.state.learning_rates['learning_rate'] = self._optimizer.get_lr()
            self.state.batch_index = idx

            self._trigger_callbacks('on_train_batch_begin')

            inputs = _to_device(inputs, self.device)
            labels = _to_device(labels, self.device)

            embeddings = self.embed_model(inputs)
            loss = self._loss(embeddings, labels)

            self._optimizer.clear_grad()
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

        :param args: Arguments to pass to ``paddle.save`` function
        :param kwargs: Keyword arguments to pass to ``paddle.save`` function
        """

        paddle.save(self.embed_model.state_dict(), *args, **kwargs)


def get_device(device: str):
    """Get Paddle compute device.

    :param device: device name
    """

    # translate our own alias into framework-compatible ones
    if device == 'cuda':
        return paddle.CUDAPlace(0)
    elif device == 'cpu':
        return paddle.CPUPlace()
    else:
        raise ValueError(
            f'Device {device} not recognized, only "cuda" and "cpu" are accepted'
        )
