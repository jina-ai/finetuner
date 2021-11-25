from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Union

import paddle
from paddle import nn
from paddle.fluid.dataloader.dataloader_iter import default_collate_fn
from paddle.io import DataLoader
from paddle.optimizer import Adam, Optimizer

from . import losses
from .datasets import PaddleClassDataset, PaddleSessionDataset
from ..base import BaseTuner
from ..state import TunerState
from ... import __default_tag_key__

if TYPE_CHECKING:
    from ...helper import DocumentSequence, PreprocFnType, CollateFnType


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


class PaddleTuner(BaseTuner[nn.Layer, DataLoader, Optimizer]):
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
            dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn_all
        )

        return data_loader

    def _get_default_optimizer(self, learning_rate: float) -> Optimizer:
        """Get the default optimizer (Adam), if none was provided by user."""

        return Adam(
            parameters=self._embed_model.parameters(), learning_rate=learning_rate
        )

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
            self.state.batch_index = idx
            self._trigger_callbacks('on_train_batch_begin')

            inputs = _to_device(inputs, self.device)
            labels = _to_device(labels, self.device)

            embeddings = self.embed_model(inputs)
            loss = self._loss(embeddings, labels)

            self._optimizer.clear_grad()
            loss.backward()
            self._optimizer.step()

            self.state.current_loss = loss.item()

            self._trigger_callbacks('on_train_batch_end')

    def fit(
        self,
        train_data: 'DocumentSequence',
        eval_data: Optional['DocumentSequence'] = None,
        epochs: int = 10,
        batch_size: int = 256,
        num_items_per_class: Optional[int] = None,
        device: str = 'cpu',
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
        **kwargs,
    ):
        """Finetune the model on the training data.
        :param train_data: Data on which to train the model
        :param eval_data: Data on which to evaluate the model at the end of each epoch
        :param preprocess_fn: A pre-processing function. It should take as input the
            content of an item in the dataset and return the pre-processed content
        :param collate_fn: The collation function to merge the content of individual
            items into a batch. Should accept a list with the content of each item,
            and output a tensor (or a list/dict of tensors) that feed directly into the
            embedding model
        :param epochs: Number of epochs to train the model
        :param batch_size: The batch size to use for training and evaluation
        :param num_items_per_class: Number of items from a single class to include in
            the batch. Only relevant for class datasets
        :param device: The device to which to move the model. Supported options are
            ``"cpu"`` and ``"cuda"`` (for GPU)
        """
        # Get dataloaders
        train_dl = self._get_data_loader(
            train_data,
            batch_size=batch_size,
            num_items_per_class=num_items_per_class,
            shuffle=True,
            preprocess_fn=preprocess_fn,
            collate_fn=collate_fn,
        )
        if eval_data:
            eval_dl = self._get_data_loader(
                eval_data,
                batch_size=batch_size,
                num_items_per_class=num_items_per_class,
                shuffle=False,
                preprocess_fn=preprocess_fn,
                collate_fn=collate_fn,
            )

        # Place model on device
        self.device = get_device(device)
        self._embed_model.to(device=self.device)

        self.state = TunerState(num_epochs=epochs)
        self._trigger_callbacks('on_fit_begin')

        for epoch in range(epochs):

            # Setting here as re-shuffling can change number of batches
            self.state.epoch = epoch
            self.state.num_batches_train = len(train_dl)

            self._trigger_callbacks('on_epoch_begin')

            self._trigger_callbacks('on_train_begin')
            self._train(train_dl)
            self._trigger_callbacks('on_train_end')

            if eval_data:
                self.state.num_batches_val = len(eval_dl)
                self._trigger_callbacks('on_val_begin')
                self._eval(eval_dl)
                self._trigger_callbacks('on_val_end')

            self._trigger_callbacks('on_epoch_end')

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
