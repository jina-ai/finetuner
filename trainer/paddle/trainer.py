from typing import Optional, Union, Iterator, Callable
from pathlib import Path
import time
import numpy as np
import paddle
from paddle import nn

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

from jina import Document, DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap

from ..base import BaseTrainer
from .models.siamese import SiameseNet
from .helper import create_dataloader
from .nn import head_layers

from tqdm import tqdm


class PaddleTrainer(BaseTrainer):
    def __init__(
        self,
        base_model: Union['nn.Layer', str],
        arity: Optional[int] = 2,
        head_layer: Union['nn.Layer', str, None] = 'CosineLayer',
        loss_fn: Union['nn.Layer', str, None] = None,
        checkpoint_dir: Optional[Union[Path, str]] = None,
        **kwargs,
    ):
        super().__init__(base_model, arity, head_layer, loss_fn, **kwargs)

        # if not isinstance(self.base_model, paddle.nn.Layer):
        #     raise TypeError(f'The model {self.base_model.__name__} is not a `paddle.nn.Layer` object.')

        self._optimizer = paddle.optimizer.RMSProp(
            learning_rate=0.01, parameters=self.wrapped_model.parameters()
        )

        if checkpoint_dir and isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)

        if checkpoint_dir and not checkpoint_dir.exists():
            raise ValueError(f'The checkpoint dir {checkpoint_dir} does not exists')

        self.checkpoint_dir = checkpoint_dir if checkpoint_dir else Path(f'checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.current_epoch = 0

        self._load_checkpoint()

    @cached_property
    def base_model(self) -> 'nn.Layer':
        return self._base_model

    @property
    def base_model_name(self):
        return type(self.base_model).__name__

    @property
    def arity(self) -> int:
        return self._arity

    @cached_property
    def head_layer(self) -> 'nn.Layer':
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)()
        elif isinstance(self._head_layer, nn.Layer):
            return self._head_layer

    @property
    def head_layer_name(self):
        return type(self.head_layer).__name__

    @property
    def optimizer(self) -> 'paddle.optimizer.Optimizer':
        return self._optimizer

    @property
    def optimizer_name(self):
        return type(self.optimizer).__name__

    @property
    def loss(self) -> str:
        return self._loss

    @cached_property
    def wrapped_model(self) -> 'nn.Layer':
        if self.base_model is None:
            raise ValueError(f'base_model is not set')
        if self.arity == 2:
            return SiameseNet(
                base_model=self.base_model,
                head_layer=self.head_layer,
                loss_fn=self.loss,
            )
        else:
            raise NotImplementedError

    def create_dataset(self, doc_array, **kwargs):
        class _Dataset(paddle.io.Dataset):
            def __init__(self, docs):
                self._img_shape = [1, 28, 28]
                self._data = []
                for d in tqdm(docs, desc='building dataset'):
                    d_blob = d.blob.reshape(self._img_shape)
                    for m in d.matches:
                        example = (
                            d_blob.astype('float32'),
                            m.blob.reshape(self._img_shape).astype('float32'),
                        ), np.float32(m.tags['trainer']['label'])
                        self._data.append(example)

            def __getitem__(self, idx):
                return self._data[idx]

            def __len__(self):
                return len(self._data)

        return _Dataset(doc_array() if callable(doc_array) else doc_array)

    def fit(
        self,
        train_data: Union[
            DocumentArray,
            DocumentArrayMemmap,
            Iterator[Document],
            Callable[..., Iterator[Document]],
        ],
        dev_data=None,
        batch_size: int = 256,
        shuffle: bool = True,
        epochs: int = 10,
        use_gpu: bool = False,
        **kwargs,
    ):
        if use_gpu:
            paddle.set_device('gpu')

        train_loader = create_dataloader(
            self.create_dataset(train_data),
            mode='train',
            batch_size=batch_size,
            shuffle=shuffle,
        )

        self.wrapped_model.train()
        for i in range(epochs):
            self.current_epoch += 1
            losses = []
            accs = []
            for batch_id, batch_data in enumerate(train_loader()):
                # forward step
                loss, acc = self.wrapped_model.training_step(batch_data, batch_id)
                avg_loss = paddle.mean(loss)

                # backward step
                avg_loss.backward()

                # TODO: gradient accumulate

                # update parameters
                self.optimizer.step()

                # clean gradients
                self.optimizer.clear_grad()

                losses.append(avg_loss.numpy()[0])
                accs.append(acc.numpy()[0])

                # if batch_id % 100 == 0:
                #     print(
                #         "=> Epoch {} step {}, Loss = {:}, Acc = {:}".format(
                #             epoch, batch_id, avg_loss.numpy(), acc.numpy()
                #         )
                #     )

            self._save_checkpoint()
            print(
                f'Epoch {self.current_epoch}, Loss = {sum(losses) / len(losses)}, Acc = {sum(accs) / len(accs)}'
            )

            # evaluate (TODO)

    def _load_checkpoint(self):
        """Load checkpoint and state dict"""
        max_epoch = -1
        for file_path in self.checkpoint_dir.glob('epoch_*'):
            _epoch = file_path.name.split('_')[-1]
            if not _epoch.isdigit():
                continue

            max_epoch = max(max_epoch, int(_epoch))

        if max_epoch == -1:
            print('[warning] model checkpoint not found, start from scratch...')
            return

        self.current_epoch = max_epoch

        self.load(self.checkpoint_dir / f'epoch_{max_epoch}')

    def _save_checkpoint(self):
        """Save model checkpoint and state dict"""
        save_dir = self.checkpoint_dir / f'epoch_{self.current_epoch}'
        print(f'=> saving model checkpoint to {save_dir}')
        self.save(save_dir)

    def load(self, load_dir: Union[Path, str]):
        """load model"""
        if isinstance(load_dir, str):
            load_dir = Path(load_dir)
        print(f'=> loading model checkpoint from {load_dir}')
        # load base model checkpoint
        state_dict = paddle.load(str(load_dir / f'{self.base_model_name}.pdparams'))
        self.base_model.set_state_dict(state_dict)

        # load head layer checkpoint
        state_dict = paddle.load(str(load_dir / f'{self.head_layer_name}.pdparams'))
        self.head_layer.set_state_dict(state_dict)

        # load optimizer checkpoint
        state_dict = paddle.load(str(load_dir / f'{self.optimizer_name}.pdopt'))
        self.optimizer.set_state_dict(state_dict)

    def save(self, save_dir: Union[Path, str]):
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        paddle.save(
            self.base_model.state_dict(),
            str(save_dir / f'{self.base_model_name}.pdparams'),
        )
        paddle.save(
            self.head_layer.state_dict(),
            str(save_dir / f'{self.head_layer_name}.pdparams'),
        )
        paddle.save(
            self.optimizer.state_dict(), str(save_dir / f'{self.optimizer_name}.pdopt')
        )
