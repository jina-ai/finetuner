from typing import Dict, Optional, Union

import torch
from jina.logging.profile import ProgressBar
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from . import losses, datasets
from ..base import BaseTuner, BaseLoss
from ..dataset.helper import get_dataset
from ..logger import LogGenerator
from ...helper import DocumentArrayLike
from ..stats import TunerStats


class PytorchTuner(BaseTuner):
    def _get_loss(self, loss: Union[BaseLoss, str]):
        if isinstance(loss, str):
            return getattr(losses, loss)()
        elif isinstance(loss, BaseLoss):
            return loss

    def _get_data_loader(self, inputs, batch_size: int, shuffle: bool):
        ds = get_dataset(datasets, self.arity)
        return DataLoader(
            dataset=ds(inputs=inputs, catalog=self._catalog),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _get_optimizer(
        self, optimizer: str, optimizer_kwargs: Optional[dict], learning_rate: float
    ) -> Optimizer:
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

    def _eval(self, data, description: str = 'Evaluating', train_log: str = ''):
        self._embed_model.eval()

        losses = []
        log_generator = LogGenerator('E', losses, train_log)

        with ProgressBar(
            description, message_on_done=log_generator, total_length=self._eval_data_len
        ) as p:
            self._eval_data_len = 0
            for inputs, label in data:
                # Inputs come as tuples or triplets
                inputs = [inpt.to(self.device) for inpt in inputs]
                label = label.to(self.device)

                with torch.inference_mode():
                    embeddings = [self._embed_model(inpt) for inpt in inputs]
                    loss = self._loss(embeddings, label)

                losses.append(loss.item())

                p.update(message=log_generator())
                self._eval_data_len += 1

        return losses

    def _train(self, data, optimizer: Optimizer, description: str):

        self._embed_model.train()

        losses = []

        log_generator = LogGenerator('T', losses)
        with ProgressBar(
            description,
            message_on_done=log_generator,
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

                losses.append(loss.item())

                p.update(message=log_generator())
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
    ) -> TunerStats:
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

        stats = TunerStats()

        for epoch in range(epochs):
            _data = self._get_data_loader(
                inputs=train_data, batch_size=batch_size, shuffle=False
            )
            lt = self._train(
                _data,
                _optimizer,
                description=f'Epoch {epoch + 1}/{epochs}',
            )
            stats.add_train_loss(lt)

            if eval_data:
                _data = self._get_data_loader(
                    inputs=eval_data, batch_size=batch_size, shuffle=False
                )

                le = self._eval(_data, train_log=LogGenerator('T', lt)())
                stats.add_eval_loss(le)
                stats.add_eval_metric(self.get_metrics(eval_data))

            stats.print_last()
        return stats

    def get_embeddings(self, data: DocumentArrayLike):
        blobs = data.blobs

        tensor = torch.tensor(blobs, device=self.device)
        with torch.inference_mode():
            embeddings = self.embed_model(tensor)
            for doc, embed in zip(data, embeddings):
                doc.embedding = embed.cpu().numpy()

    def save(self, *args, **kwargs):
        torch.save(self.embed_model.state_dict(), *args, **kwargs)
