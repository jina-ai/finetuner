from pathlib import Path
from typing import Union, Iterator, Callable

import numpy as np
import paddle
from jina import Document, DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap
from paddle import nn
from tqdm import tqdm

import head_layers
from .helper import create_dataloader
from ..base import BaseTrainer


class _ArityModel(nn.Layer):
    """The helper class to copy the network for multi-inputs. """

    def __init__(self, base_model: nn.Layer):
        super().__init__()
        self._base_model = base_model

    def forward(self, *args):
        return tuple(self._base_model(a) for a in args)


class PaddleTrainer(BaseTrainer):

    @property
    def head_layer(self) -> 'nn.Layer':
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)()
        elif isinstance(self._head_layer, nn.Layer):
            return self._head_layer

    @property
    def wrapped_model(self) -> 'nn.Layer':
        if self.base_model is None:
            raise ValueError(f'base_model is not set')

        return self.head_layer(_ArityModel(self.base_model))  # wrap with head layer

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
        model = self.wrapped_model
        if use_gpu:
            paddle.set_device('gpu')

        train_loader = create_dataloader(
            self.create_dataset(train_data),
            mode='train',
            batch_size=batch_size,
            shuffle=shuffle,
        )

        optimizer = paddle.optimizer.RMSProp(
            learning_rate=0.01, parameters=model.parameters()
        )

        loss_fn = self.head_layer.default_loss

        for epoch in range(epochs):
            model.train()

            losses = []
            accs = []
            for (l_input, r_input), label in train_loader:
                head_value = model(l_input, r_input)
                loss = loss_fn(head_value, label)

                # forward step
                optimizer.clear_grad()
                # backward step
                loss.backward()
                # update parameters
                optimizer.step()

                losses.append(loss.numpy())

            print(
                f'Epoch {epoch}, Loss = {sum(losses) / len(losses)}'
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

        self.load_checkpoint(self.checkpoint_dir / f'epoch_{max_epoch}')

    def _save_checkpoint(self):
        """Save model checkpoint and state dict"""
        save_dir = self.checkpoint_dir / f'epoch_{self.current_epoch}'
        self.save_checkpoint(save_dir)

    def load_checkpoint(self, load_dir: Union[Path, str]):
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

    def save_checkpoint(self, save_dir: Union[Path, str]):
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        print(f'=> saving model checkpoint to {save_dir}')
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

    def save(self, save_path: str):
        paddle.save(self.base_model, save_path)
