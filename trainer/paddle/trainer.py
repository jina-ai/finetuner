from typing import Optional, Union, Iterator, Callable
import functools
import numpy as np
import paddle
from paddle import nn

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
        loss: Union['nn.Layer', str, None] = None,

        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(base_model, arity, head_layer, loss, **kwargs)

        self._wrapped_model = self.wrapped_model()

        # self._optimizer = paddle.optimizer.RMSProp(
        #     learning_rate=0.1, parameters=self._wrapped_model.parameters()
        # )
        self._optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=self._wrapped_model.parameters())
        if use_gpu:
            paddle.set_device('gpu')

        self.kwargs = kwargs

    @property
    def base_model(self) -> 'nn.Layer':
        return self._base_model

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def head_layer(self) -> 'nn.Layer':
        return self._head_layer

    @property
    def head_layer(self) -> 'nn.Layer':
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)()
        elif isinstance(self._head_layer, nn.Layer):
            return self._head_layer

    @property
    def loss(self) -> str:
        return self._loss

    # @functools.cached_property
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
        **kwargs,
    ):
        train_loader = create_dataloader(
            self.create_dataset(train_data),
            mode='train',
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self._wrapped_model.train()
        for epoch in range(epochs):
            losses = []
            accs = []
            for batch_id, batch_data in enumerate(train_loader()):
                # forward step
                loss, acc = self._wrapped_model.training_step(batch_data, batch_id)
                avg_loss = paddle.mean(loss)

                # backward step
                avg_loss.backward()

                # TODO: gradient accumulate

                # update parameters
                self._optimizer.step()

                # clean gradients
                self._optimizer.clear_grad()

                losses.append(avg_loss.numpy()[0])
                accs.append(acc.numpy()[0])

                # if batch_id % 100 == 0:
                #     print(
                #         "=> Epoch {} step {}, Loss = {:}, Acc = {:}".format(
                #             epoch, batch_id, avg_loss.numpy(), acc.numpy()
                #         )
                #     )

            self.save(f'checkpoints/epoch_{epoch}.pd')
            print(
                f'Epoch {epoch}, Loss = {sum(losses) / len(losses)}, Acc = {sum(accs) / len(accs)}'
            )

            # evaluate (TODO)

    def save(self, target_filepath: str):
        model_dict = self.base_model.state_dict()
        paddle.save(model_dict, target_filepath)
