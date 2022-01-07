from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Union

import paddle
from paddle import nn
from paddle.fluid.dataloader.dataloader_iter import default_collate_fn
from paddle.io import DataLoader
from paddle.optimizer import Adam, Optimizer
from paddle.optimizer.lr import LRScheduler

from ... import __default_tag_key__
from ...excepts import DimensionMismatchException
from ..base import BaseTuner
from ..dataset.datasets import InstanceDataset
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
            if len(data[0].matches) > 0:
                dataset = PaddleSessionDataset(data, preprocess_fn=preprocess_fn)
            else:
                dataset = InstanceDataset(data, preprocess_fn=preprocess_fn)

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

    def _attach_projection_head(
        self,
        output_dim: Optional[int] = 128,
        num_layers: Optional[int] = 3,
    ):
        """Attach a projection head on top of the embed model for self-supervised learning.
        Calling this function will modify :attr:`self._embed_model` in place.

        :param output_dim: The output dimensionality of the projection, default 128, recommend 32, 64, 128, 256.
        :param num_layers: Number of layers of the projection head, default 3, recommend 2, 3.
        """
        # interpret embed model output shape
        rand_input = paddle.cast(paddle.rand(self._input_size), self._input_dtype)
        output = self._embed_model(paddle.unsqueeze(rand_input, axis=0))
        if not len(output.shape) == 2:
            raise DimensionMismatchException(
                f'Expected input shape is 2d, got {len(output.size())}.'
            )
        projection_head = _ProjectionHead(output.shape[1], output_dim, num_layers)
        embed_model_with_projection_head = nn.Sequential()
        embed_model_with_projection_head.add_sublayer('embed_model', self._embed_model)
        embed_model_with_projection_head.add_sublayer(
            'projection_head', projection_head
        )
        self._embed_model = embed_model_with_projection_head

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

        # If self-supervised, add projection head.
        if isinstance(train_dl.dataset, InstanceDataset):
            self._attach_projection_head(output_dim=128, num_layers=3)

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

        # If self-supervised, drop projection head
        if isinstance(train_dl.dataset, InstanceDataset):
            del self._embed_model.projection_head

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


class _ProjectionHead(nn.Layer):
    """Projection head used internally for self-supervised training.
    It is (by default) a simple 3-layer MLP to be attached on top of embedding model only for training purpose.
    After training, it should be cut-out from the embedding model.
    """

    EPSILON = 1e-5

    def __init__(self, in_features: int, output_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.head_layers = nn.LayerList()
        is_last_layer = False
        for idx in range(num_layers):
            if idx == num_layers - 1:
                is_last_layer = True
            if not is_last_layer:
                self.head_layers.append(
                    nn.Linear(
                        in_features=in_features,
                        out_features=in_features,
                        bias_attr=False,
                    )
                )
                self.head_layers.append(
                    nn.BatchNorm1D(num_features=in_features, epsilon=self.EPSILON)
                )
                self.head_layers.append(nn.ReLU())
            else:
                self.head_layers.append(
                    nn.Linear(
                        in_features=in_features,
                        out_features=output_dim,
                        bias_attr=False,
                    )
                )

    def forward(self, x):
        for layer in self.head_layers:
            x = layer(x)
        return x
