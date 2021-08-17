from typing import Optional
import numpy as np
import paddle
import paddle.fluid as fluid

from ..base import BaseTrainer

class PaddleTrainer(BaseTrainer):
    def __init__(self, model: 'nn.Layer', loss_fn, optimizer, init_lr: float = 0.001, **kwargs):
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer if optimizer else paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
        self._init_lr = init_lr

        self.kwargs = kwargs

    def fit(self, train_loader, dev_loader, epochs: int = 1, **kwargs):
        # Starts training and evaluating.
        for epoch in range(epochs):
            for batch_id, batch_data in enumerate(train_loader):
                # img = data[0]
                # label = data[1]

                # 网络正向执行
                pred, acc = self._model(**batch_data)

                loss = self._loss_fn(pred, label)
                avg_loss = paddle.mean(loss)

                avg_loss.backward()

                # 参数更新
                self._optimizer.step()
                # 将本次计算的梯度值清零，以便进行下一次迭代和梯度更新
                self._optimizer.clear_grad()

                # 输出对应epoch、batch_id下的损失值，预测精确度
                if batch_id % 100 == 0:
                    print("Epoch {} step {}, Loss = {:}, Accuracy = {:}".format(
                        epoch, batch_id, avg_loss.numpy(), acc))

            # evaluate

    def save(self, target_filepath: str):
        model_dict = self._model.state_dict()
        paddle.save(model_dict, target_filepath)


