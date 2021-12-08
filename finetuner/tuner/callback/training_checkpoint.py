import os
from typing import TYPE_CHECKING

import pickle
import paddle
import torch
from jina.logging.logger import JinaLogger


from finetuner.helper import get_framework
from .base import BaseCallback


if TYPE_CHECKING:
    from ..base import BaseTuner


class TrainingCheckpoint(BaseCallback):
    """
    Callback to save model at every epoch
    `TrainingModelCheckpoint` is used in conjunction with training
    using `finetuner.fit()`

    An option this callback provides include:
    - Definition of 'best'; which quantity to monitor and whether it should be
        maximized or minimized.
    """

    def __init__(
        self,
        save_dir: str = None,
    ):
        """
        :param save_dir: string or `PathLike`, path to save the model file.
        """
        self._logger = JinaLogger(self.__class__.__name__)
        self._save_dir = save_dir
        if not save_dir:
            raise ValueError(
                '``save_dir`` parameter is mandatory. Pass it in parameters'
            )

    def on_epoch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the training epoch.
        """
        self._save_model_framework(tuner)

    def _save_model_framework(self, tuner):
        """
        Saves the model depending on its framework.
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
            state = {
                'epoch': tuner.state.epoch + 1,
                'state_dict': tuner.embed_model.state_dict(),
                'optimizer': tuner._optimizer.state_dict(),
            }

            torch.save(state, f=self._get_file_path(tuner))
        elif get_framework(tuner.embed_model) == 'paddle':
            state = {
                'epoch': tuner.state.epoch + 1,
                'state_dict': tuner.embed_model.state_dict(),
                'optimizer': tuner._optimizer.state_dict(),
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
            tuner.load(fp)
            with open(os.path.join(fp, 'saved_state.pkl'), 'rb') as f:
                loaded_state = pickle.load(f)
            tuner.state.epoch = loaded_state['epoch']
        elif get_framework(tuner.embed_model) == 'torch':
            checkpoint = torch.load(fp)
            tuner._embed_model.load_state_dict(checkpoint['state_dict'])
            tuner._optimizer.load_state_dict(checkpoint['optimizer'])
            tuner.state.epoch = checkpoint['epoch']
        elif get_framework(tuner.embed_model) == 'paddle':
            checkpoint = paddle.load(fp)
            tuner._embed_model.set_state_dict(checkpoint['state_dict'])
            tuner._optimizer.set_state_dict(checkpoint['optimizer'])
            tuner.state.epoch = checkpoint['epoch']
