import os
import pickle
from typing import TYPE_CHECKING

import keras
import paddle
import torch
from jina.logging.logger import JinaLogger

from finetuner.helper import get_framework

from .base import BaseCallback

if TYPE_CHECKING:
    from ..base import BaseTuner


class TrainingCheckpoint(BaseCallback):
    """
    Callback to save model at every epoch or the last K epochs
    """

    def __init__(self, save_dir: str, last_k_epochs: int = None):
        """
        :param save_dir: string, path to save the model file.
        :param last_k_epochs: this parameter is an integer. It allows you yo save
            only at the last k epochs and not on every epoch.
            For example if `last_k_epochs = 3` and the total number of epochs is 10
            we'll save only on epochs 8,9 and 10.
        """
        self._logger = JinaLogger(self.__class__.__name__)
        self._save_dir = save_dir
        self._last_k_epochs = last_k_epochs
        if not save_dir:
            raise ValueError(
                '``save_dir`` parameter is mandatory. Pass it in parameters'
            )

    def on_epoch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the training epoch.
        """
        if self._last_k_epochs == None:
            self._save_model_framework(tuner)
            self._logger.logger.info(
                f'Model trained for {tuner.state.epoch+1} epochs is saved!'
            )
        else:
            if tuner.state.epoch >= tuner.state.num_epochs - self._last_k_epochs:
                self._save_model_framework(tuner)
                self._logger.logger.info(
                    f'Model trained for {tuner.state.epoch+1} epochs is saved!'
                )

    def _save_model_framework(self, tuner):
        """
        Saves the model weights, optimizer, scheduler and epoch
        depending on its framework.
        """
        if get_framework(tuner.embed_model) == 'keras':
            tuner.save(
                filepath=self._get_file_path(tuner),
            )
            state = {
                'epoch': tuner.state.epoch + 1,
            }
            with open(
                os.path.join(self._get_file_path(tuner), 'saved_state.pkl'), 'wb'
            ) as f:
                pickle.dump(state, f)
        elif get_framework(tuner.embed_model) == 'torch':
            if tuner._scheduler and hasattr(tuner._scheduler, 'state_dict'):
                state = {
                    'epoch': tuner.state.epoch + 1,
                    'state_dict': tuner.embed_model.state_dict(),
                    'optimizer': tuner._optimizer.state_dict(),
                    'scheduler': tuner._scheduler.state_dict(),
                }
            else:
                state = {
                    'epoch': tuner.state.epoch + 1,
                    'state_dict': tuner.embed_model.state_dict(),
                    'optimizer': tuner._optimizer.state_dict(),
                    'scheduler': None,
                }

            torch.save(state, f=self._get_file_path(tuner))
        elif get_framework(tuner.embed_model) == 'paddle':
            if tuner._scheduler and hasattr(tuner._scheduler, 'state_dict'):
                state = {
                    'epoch': tuner.state.epoch + 1,
                    'state_dict': tuner.embed_model.state_dict(),
                    'optimizer': tuner._optimizer.state_dict(),
                    'scheduler': tuner._scheduler.state_dict(),
                }
            else:
                state = {
                    'epoch': tuner.state.epoch + 1,
                    'state_dict': tuner.embed_model.state_dict(),
                    'optimizer': tuner._optimizer.state_dict(),
                    'scheduler': None,
                }

            paddle.save(state, path=self._get_file_path(tuner))

    def _get_file_path(self, tuner):
        """
        Returns the file path for checkpoint.
        """

        file_path = os.path.join(
            self._save_dir,
            'saved_model_epoch_{:02d}'.format(tuner.state.epoch + 1),
        )
        return file_path

    @staticmethod
    def load_model(tuner: 'BaseTuner', fp: str):
        """
        Loads the model and tuner state
        """
        if get_framework(tuner.embed_model) == 'keras':
            tuner._embed_model = keras.models.load_model(fp)
            with open(os.path.join(fp, 'saved_state.pkl'), 'rb') as f:
                loaded_state = pickle.load(f)
            tuner.state.epoch = loaded_state['epoch']
        elif get_framework(tuner.embed_model) == 'torch':
            checkpoint = torch.load(fp)
            tuner._embed_model.load_state_dict(checkpoint['state_dict'])
            tuner._optimizer.load_state_dict(checkpoint['optimizer'])
            if tuner._scheduler and hasattr(tuner._scheduler, 'state_dict'):
                tuner._scheduler.load_state_dict(checkpoint['scheduler'])
            tuner.state.epoch = checkpoint['epoch']
        elif get_framework(tuner.embed_model) == 'paddle':
            checkpoint = paddle.load(fp)
            tuner._embed_model.set_state_dict(checkpoint['state_dict'])
            tuner._optimizer.set_state_dict(checkpoint['optimizer'])
            if tuner._scheduler and hasattr(tuner._scheduler, 'state_dict'):
                tuner._scheduler.set_state_dict(checkpoint['scheduler'])
            tuner.state.epoch = checkpoint['epoch']
