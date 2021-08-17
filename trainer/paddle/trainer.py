from typing import Optional
import numpy as np
import paddle
import paddle.fluid as fluid

# from ..base import BaseTrainer

class PaddleTrainer:
    def __init__(self, model: 'nn.Layer', optimizer = None, init_lr: float = 0.001, **kwargs):
        self._model = model
        self._optimizer = optimizer if optimizer else paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
        self._init_lr = init_lr

        self.kwargs = kwargs

    def fit(self, train_loader, dev_loader = None, epochs: int = 1, **kwargs):
        # Starts training and evaluating.
        for epoch in range(epochs):
            for batch_id, batch_data in enumerate(train_loader):
                loss = self._model.training_step(batch_data, batch_id)
                avg_loss = paddle.mean(loss)

                # backward gradient
                avg_loss.backward()

                # update parameters
                self._optimizer.step()

                # clean gradients
                self._optimizer.clear_grad()


                if batch_id % 100 == 0:
                    print("Epoch {} step {}, Loss = {:}".format(
                        epoch, batch_id, avg_loss.numpy()))

            # evaluate (TODO)

    def save(self, target_filepath: str):
        model_dict = self._model.state_dict()
        paddle.save(model_dict, target_filepath)

