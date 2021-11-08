from typing import Callable, Dict, Mapping, Optional, Sequence, Union

import torch
from jina.logging.profile import ProgressBar
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from ... import __default_tag_key__
from ...helper import DocumentSequence
from ..base import BaseMiner, BaseTuner
from ..summary import ScalarSequence, Summary
from . import losses
from .datasets import PytorchClassDataset, PytorchSessionDataset


def _to_device(
    inputs: Union[torch.Tensor, Mapping[str, torch.Tensor]], device: torch.device
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    elif isinstance(inputs, Mapping):
        return {k: v.to(device) for k, v in inputs.items()}
    elif isinstance(inputs, Sequence):
        return [x.to(device) for x in inputs]


class PytorchTuner(BaseTuner[nn.Module, DataLoader, Optimizer]):
    def _get_loss(self, loss: Union[nn.Module, str]) -> nn.Module:
        """Get the loss layer."""
        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, nn.Module):
            return loss

    def _get_dataset(
        self, data: DocumentSequence, preprocess_fn: Optional[Callable]
    ) -> Union[PytorchClassDataset, PytorchSessionDataset]:
        if __default_tag_key__ in data[0].tags:
            dataset = PytorchClassDataset(data, preprocess_fn=preprocess_fn)
        else:
            dataset = PytorchSessionDataset(data, preprocess_fn=preprocess_fn)

        return dataset

    def _get_optimizer(
        self, optimizer: str, optimizer_kwargs: Optional[dict], learning_rate: float
    ) -> Optimizer:
        """Get the optimizer for training."""

        params = self._embed_model.parameters()
        optimizer_kwargs = self._get_optimizer_kwargs(optimizer, optimizer_kwargs)

        if optimizer == 'adam':
            return torch.optim.Adam(
                params,
                lr=learning_rate,
                betas=(optimizer_kwargs['beta_1'], optimizer_kwargs['beta_2']),
                eps=optimizer_kwargs['epsilon'],
            )
        elif optimizer == 'rmsprop':
            return torch.optim.RMSprop(
                params,
                lr=learning_rate,
                alpha=optimizer_kwargs['rho'],
                centered=optimizer_kwargs['centered'],
                eps=optimizer_kwargs['epsilon'],
                momentum=optimizer_kwargs['momentum'],
            )
        elif optimizer == 'sgg':
            return torch.optim.SGD(
                params,
                lr=learning_rate,
                momentum=optimizer_kwargs['momentum'],
                nesterov=optimizer_kwargs['nesterov'],
            )

    def _eval(
        self,
        data: DataLoader,
        description: str = 'Evaluating',
        train_loss: Optional[ScalarSequence] = None,
    ) -> ScalarSequence:
        """Evaluate the model on given labeled data"""

        self._embed_model.eval()

        _summary = ScalarSequence('Eval Loss')

        with ProgressBar(
            description,
            message_on_done=lambda: f'{train_loss} | {_summary}',
            total_length=len(data),
        ) as p:
            for inputs, labels in data:
                inputs = _to_device(inputs, self.device)
                labels = _to_device(labels, self.device)

                # Can not use inference mode due to (when having too many triplets)
                # https://github.com/pytorch/pytorch/issues/60539
                with torch.no_grad():
                    embeddings = self.embed_model(inputs)
                    dists = losses.get_distance(embeddings, self._loss.distance)

                    mined_tuples = self._miner.mine(labels, dists)
                    loss = self._loss(embeddings, mined_tuples)

                _summary += loss.item()

                p.update(message=str(_summary))

        return _summary

    def _train(
        self, data: DataLoader, optimizer: Optimizer, description: str
    ) -> ScalarSequence:
        """Train the model on given labeled data"""

        self._embed_model.train()

        _summary = ScalarSequence('Train Loss')
        with ProgressBar(
            description,
            message_on_done=_summary.__str__,
            final_line_feed=False,
            total_length=len(data),
        ) as p:
            for inputs, labels in data:
                inputs = _to_device(inputs, self.device)
                labels = _to_device(labels, self.device)

                embeddings = self.embed_model(inputs)
                with torch.inference_mode():
                    dists = losses.get_distance(embeddings, self._loss.distance)

                mined_tuples = self._miner.mine(labels, dists)
                loss = self._loss(embeddings, mined_tuples)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _summary += loss.item()
                p.update(message=str(_summary))

        return _summary

    def fit(
        self,
        train_data: DocumentSequence,
        eval_data: Optional[DocumentSequence] = None,
        preprocess_fn: Optional[Callable] = None,
        epochs: int = 10,
        miner: Optional[BaseMiner] = None,
        batch_size: int = 256,
        num_items_per_class: int = 4,
        collate_fn: Optional[Callable] = None,
        learning_rate: float = 1e-3,
        optimizer: str = 'adam',
        optimizer_kwargs: Optional[Dict] = None,
        device: str = 'cpu',
        **kwargs,
    ) -> Summary:
        """Finetune the model on the training data.

        :param train_data: Data on which to train the model
        :param eval_data: Data on which to evaluate the model at the end of each epoch
        :param preprocess_fn: A pre-processing function. It should take as input the
            content of an item in the dataset and return the pre-processed content
        :param epochs: Number of epochs to train the model
        :param batch_size: The batch size to use for training and evaluation
        :param num_items_per_class: Number of items from a single class to include in
            the batch. Only relevant for class datasets
        :param collate_fn: The collation function to merge individual items into batch
            inputs for the model (plus labels). Will be passed to torch ``DataLoader``
        :param learning_rate: Learning rate to use in training
        :param optimizer: Which optimizer to use in training. Supported
            values/optimizers are:
            - ``"adam"`` for the Adam optimizer
            - ``"rmsprop"`` for the RMSProp optimizer
            - ``"sgd"`` for the SGD optimizer with momentum
        :param optimizer_kwargs: Keyword arguments to pass to the optimizer. The
            supported arguments, togethere with their default values, are:
            - ``"adam"``:  ``{'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08}``
            - ``"rmsprop"``::

                {
                    'rho': 0.99,
                    'momentum': 0.0,
                    'epsilon': 1e-08,
                    'centered': False,
                }

            - ``"sgd"``: ``{'momentum': 0.0, 'nesterov': False}``
        :param device: The device to which to move the model. Supported options are
            ``"cpu"`` and ``"cuda"`` (for GPU)
        """
        # Get dataloaders
        train_dataset = self._get_dataset(train_data, preprocess_fn)
        train_batch_sampler = self.get_batch_sampler(
            train_dataset, batch_size, num_items_per_class, True
        )
        train_dl = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
        )
        if eval_data:
            eval_dataset = self._get_dataset(train_data, preprocess_fn)
            eval_batch_sampler = self.get_batch_sampler(
                eval_dataset, batch_size, num_items_per_class, False
            )
            eval_dl = DataLoader(
                dataset=eval_dataset,
                batch_sampler=eval_batch_sampler,
                collate_fn=collate_fn,
            )

        # Get miner
        self._miner = miner or self._loss.get_default_miner(train_dataset)

        # Place model on device
        self.device = get_device(device)
        self._embed_model = self._embed_model.to(self.device)

        # Get optimizer
        _optimizer = self._get_optimizer(optimizer, optimizer_kwargs, learning_rate)

        m_train_loss = ScalarSequence('train')
        m_eval_loss = ScalarSequence('eval')
        for epoch in range(epochs):
            lt = self._train(
                train_dl,
                _optimizer,
                description=f'Epoch {epoch + 1}/{epochs}',
            )
            m_train_loss += lt

            if eval_data:
                le = self._eval(eval_dl, train_loss=m_train_loss)
                m_eval_loss += le

        return Summary(m_train_loss, m_eval_loss)

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
