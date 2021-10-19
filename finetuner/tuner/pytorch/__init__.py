from typing import Dict, Optional, Union, List

import torch
from jina.logging.profile import ProgressBar
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from . import losses, datasets
from ..base import BaseTuner, BaseLoss
from ..dataset.helper import get_dataset
from ..summary import ScalarSummary, SummaryCollection
from ...helper import DocumentArrayLike, AnyDataLoader


class PytorchTuner(BaseTuner):
    def _get_loss(self, loss: Union[BaseLoss, str]) -> BaseLoss:
        """Get the loss layer."""
        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, BaseLoss):
            return loss

    def _get_data_loader(
        self, inputs: DocumentArrayLike, batch_size: int, shuffle: bool
    ) -> AnyDataLoader:
        """Get pytorch ``DataLoader`` data loader from the input data."""
        ds = get_dataset(datasets, self.arity)
        return DataLoader(
            dataset=ds(inputs=inputs),
            batch_size=batch_size,
            shuffle=shuffle,
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

    def _eval(
        self,
        data: AnyDataLoader,
        description: str = 'Evaluating',
        train_loss: Optional[ScalarSummary] = None,
    ) -> ScalarSummary:
        """Evaluate the model on given labeled data"""

        self._embed_model.eval()

        losses = ScalarSummary('Eval Loss')

        with ProgressBar(
            description,
            message_on_done=lambda: f'{train_loss} | {losses}',
            total_length=self._eval_data_len,
        ) as p:
            self._eval_data_len = 0
            for inputs, label in data:
                # Inputs come as tuples or triplets
                inputs = [inpt.to(self.device) for inpt in inputs]
                label = label.to(self.device)

                with torch.inference_mode():
                    embeddings = [self._embed_model(inpt) for inpt in inputs]
                    loss = self._loss(embeddings, label)

                losses += loss.item()

                p.update(message=str(losses))
                self._eval_data_len += 1

        return losses

    def _train(
        self, data: AnyDataLoader, optimizer: Optimizer, description: str
    ) -> ScalarSummary:
        """Train the model on given labeled data"""

        self._embed_model.train()

        losses = ScalarSummary('Train Loss')
        with ProgressBar(
            description,
            message_on_done=losses.__str__,
            final_line_feed=False,
            total_length=self._train_data_len,
        ) as p:
            self._train_data_len = 0
            for inputs, label in data:
                # inputs come as tuples or triplets
                inputs = [inpt.to(self.device) for inpt in inputs]
                label = label.to(self.device)

                embeddings = [self.embed_model(inpt) for inpt in inputs]
                loss = self._loss(embeddings, label)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                losses += loss.item()

                p.update(message=str(losses))
                self._train_data_len += 1
        return losses

    def fit(
        self,
        train_data: DocumentArrayLike,
        eval_data: Optional[DocumentArrayLike] = None,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        optimizer: str = 'adam',
        optimizer_kwargs: Optional[Dict] = None,
        device: str = 'cpu',
        **kwargs,
    ) -> SummaryCollection:
        """Finetune the model on the training data.

        :param train_data: Data on which to train the model
        :param eval_data: Data on which to evaluate the model at the end of each epoch
        :param epochs: Number of epochs to train the model
        :param batch_size: The batch size to use for training and evaluation
        :param learning_rate: Learning rate to use in training
        :param optimizer: Which optimizer to use in training. Supported
            values/optimizers are:
            - ``"adam"`` for the Adam optimizer
            - ``"rmsprop"`` for the RMSProp optimizer
            - ``"sgd"`` for the SGD optimizer with momentum
        :param optimizer_kwargs: Keyword arguments to pass to the optimizer. The
            supported arguments, togethere with their defailt values, are:
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
        if device == 'cpu':
            self.device = torch.device('cpu')
        elif device == 'cuda':
            self.device = torch.device('cuda')
        else:
            raise ValueError(f'Device {device} not recognized')

        # Place model on device
        self._embed_model = self._embed_model.to(self.device)

        # Get optimizer
        _optimizer = self._get_optimizer(optimizer, optimizer_kwargs, learning_rate)

        m_train_loss = ScalarSummary('train')
        m_eval_loss = ScalarSummary('eval')

        for epoch in range(epochs):
            _data = self._get_data_loader(
                inputs=train_data, batch_size=batch_size, shuffle=False
            )
            lt = self._train(
                _data,
                _optimizer,
                description=f'Epoch {epoch + 1}/{epochs}',
            )
            m_train_loss += lt

            if eval_data:
                _data = self._get_data_loader(
                    inputs=eval_data, batch_size=batch_size, shuffle=False
                )

                le = self._eval(_data, train_loss=m_train_loss)
                m_eval_loss += le

        return SummaryCollection(m_train_loss, m_eval_loss)

    def save(self, *args, **kwargs):
        """Save the embedding model.

        You need to pass the path where to save the model in either ``args`` or
        ``kwargs`` (for ``f`` key).

        :param args: Arguments to pass to ``torch.save`` function
        :param kwargs: Keyword arguments to pass to ``torch.save`` function
        """
        torch.save(self.embed_model.state_dict(), *args, **kwargs)
