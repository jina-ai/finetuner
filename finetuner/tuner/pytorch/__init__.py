from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from jina.logging.profile import ProgressBar
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from ... import __default_tag_key__
from ...helper import DocumentSequence
from ..dataset.samplers import RandomClassBatchSampler, SessionBatchSampler
from ..base import BaseTuner, BaseMiner
from ..summary import ScalarSequence, Summary
from . import losses
from .datasets import PytorchClassDataset, PytorchSessionDataset


class PytorchTuner(BaseTuner[nn.Module, DataLoader, Optimizer]):
    def _get_loss(self, loss: Union[nn.Module, str]) -> nn.Module:
        """Get the loss layer."""
        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, nn.Module):
            return loss

    def _get_data_loader(
        self,
        data: Union[DocumentSequence, PytorchClassDataset, PytorchSessionDataset],
        batch_size: int,
        num_items_per_class: int,
        shuffle: bool,
        collate_fn: Optional[Callable],
    ) -> DataLoader:
        """Get pytorch ``DataLoader`` from the input data."""

        # Infer dataset type and pick dataset and batch sampler accordingly
        if not isinstance(data, torch.utils.data.Dataset):
            if __default_tag_key__ in data[0].tags:
                dataset = PytorchClassDataset(data)
            else:
                dataset = PytorchSessionDataset(data)
        else:
            dataset = data

        if isinstance(dataset, PytorchClassDataset):
            batch_sampler = RandomClassBatchSampler(
                dataset.labels, batch_size, num_items_per_class
            )
        else:
            batch_sampler = SessionBatchSampler(dataset.labels, batch_size, shuffle)

        return DataLoader(
            dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn
        )

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
        elif optimizer == 'sgd':
            return torch.optim.SGD(
                params,
                lr=learning_rate,
                momentum=optimizer_kwargs['momentum'],
                nesterov=optimizer_kwargs['nesterov'],
            )

    @staticmethod
    def _get_distances(embeddings: torch.Tensor, distance: str) -> np.ndarray:
        """Compute the distance between items in the embeddings

        :param embeddings: An ``[N, d]`` tensor of item embeddings
        :param distance: Type of distance to compute. Supported values are
            ``"cosine"`` and ``"euclidean"``

        :return: An ``[N, N]`` matrix of item distances
        """

        with torch.inference_mode():
            if distance == 'cosine':
                emb_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                dists = 1 - torch.mm(emb_norm, emb_norm.transpose(0, 1))
            elif distance == 'euclidean':
                dists = torch.cdist(embeddings, embeddings, p=2)

        return dists.cpu().numpy()

    # def _eval(
    #     self,
    #     data: AnyDataLoader,
    #     description: str = 'Evaluating',
    #     train_loss: Optional[ScalarSequence] = None,
    # ) -> ScalarSequence:
    #     """Evaluate the model on given labeled data"""

    #     self._embed_model.eval()

    #     _summary = ScalarSequence('Eval Loss')

    #     with ProgressBar(
    #         description,
    #         message_on_done=lambda: f'{train_loss} | {_summary}',
    #         total_length=self._eval_data_len,
    #     ) as p:
    #         self._eval_data_len = 0
    #         for inputs, label in data:
    #             # Inputs come as tuples or triplets
    #             inputs = [inpt.to(self.device) for inpt in inputs]
    #             label = label.to(self.device)

    #             with torch.inference_mode():
    #                 embeddings = [self._embed_model(inpt) for inpt in inputs]
    #                 loss = self._loss(embeddings, label)

    #             _summary += loss.item()

    #             p.update(message=str(_summary))
    #             self._eval_data_len += 1

    #     return _summary

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
                # inputs come as tuples or triplets
                inputs = inputs.to(self.device)

                if isinstance(labels, list):  # If labels come from a session dataset
                    labels = [tuple(x) for x in torch.stack(labels).T.tolist()]
                else:
                    labels = labels.tolist()

                embeddings = self.embed_model(inputs)
                dists = self._get_distances(embeddings, self._loss.distance)

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
        train_data: Union[DocumentSequence, PytorchClassDataset, PytorchSessionDataset],
        # eval_data: Optional[DocumentArrayLike] = None,
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
        :param epochs: Number of epochs to train the model
        :param batch_size: The batch size to use for training and evaluation
        :param num_items_per_class: Number of items from a single class to include in
            the batch. Only relevant for class datasets
        :param collate_fn: The collation function to collate items for an individual
            batch into inputs for the model. Will be passed to torch ``DataLoader``.
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
        train_dl = self._get_data_loader(
            data=train_data,
            batch_size=batch_size,
            num_items_per_class=num_items_per_class,
            shuffle=False,
            collate_fn=collate_fn,
        )
        # eval_dl = self._get_data_loader(
        #     inputs=eval_data, batch_size=batch_size, shuffle=False
        # )

        # Get miner
        if isinstance(train_data, torch.utils.data.Dataset):
            is_session_data = isinstance(train_data, PytorchSessionDataset)
        else:
            is_session_data = __default_tag_key__ not in train_data[0].tags

        self._miner = self._get_miner(
            miner, loss=self._loss, is_session_data=is_session_data
        )

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

            # if eval_data:
            #     le = self._eval(_data, train_loss=m_train_loss)
            #     m_eval_loss += le

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
