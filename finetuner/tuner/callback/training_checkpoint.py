import logging
import os
import pickle
import shutil
from typing import TYPE_CHECKING

from ...helper import get_framework
from .base import BaseCallback

if TYPE_CHECKING:
    from ..base import BaseTuner


class TrainingCheckpoint(BaseCallback):
    """
    Callback to save model at every epoch or the last K epochs
    """

    def __init__(self, save_dir: str, last_k_epochs: int = 1, verbose: bool = False):
        """
        :param save_dir: string, path to save the model file.
        :param last_k_epochs: this parameter is an integer. Only the most
            recent k checkpoints will be kept. Older checkpoints are deleted.
        :param verbose: Whether to log notifications when a checkpoint is saved/deleted.
        """
        self._logger = logging.getLogger('finetuner.' + self.__class__.__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._save_dir = save_dir
        self._last_k_epochs = last_k_epochs
        self._saved_checkpoints = []

    def on_epoch_end(self, tuner: 'BaseTuner'):
        self._save_model(tuner)
        self._logger.info(f'Model trained for {tuner.state.epoch+1} epochs is saved!')

        if self._last_k_epochs:
            if len(self._saved_checkpoints) > self._last_k_epochs:
                if os.path.isfile(self._saved_checkpoints[0]):
                    os.remove(self._saved_checkpoints[0])
                else:
                    shutil.rmtree(self._saved_checkpoints[0])
                self._saved_checkpoints.pop(0)

                self._logger.info(
                    f'Model trained for {tuner.state.epoch+1-self._last_k_epochs}'
                    ' epochs is deleted!'
                )

    def _save_model(self, tuner: 'BaseTuner'):
        """
        Saves the model weights, optimizer, scheduler and epoch
        depending on the framework.
        """
        framework = get_framework(tuner.embed_model)

        if framework == 'keras':
            tuner.save(filepath=self._get_file_path(tuner))
            state = {'epoch': tuner.state.epoch + 1}
            with open(
                os.path.join(self._get_file_path(tuner), 'saved_state.pkl'), 'wb'
            ) as f:
                pickle.dump(state, f)

        elif framework == 'torch':
            import torch

            state = {
                'epoch': tuner.state.epoch + 1,
                'state_dict': tuner.embed_model.state_dict(),
                'optimizer': tuner._optimizer.state_dict(),
            }
            if tuner._scheduler and hasattr(tuner._scheduler, 'state_dict'):
                state['scheduler'] = tuner._scheduler.state_dict()
            torch.save(state, f=self._get_file_path(tuner))

        elif framework == 'paddle':
            import paddle

            state = {
                'epoch': tuner.state.epoch + 1,
                'state_dict': tuner.embed_model.state_dict(),
                'optimizer': tuner._optimizer.state_dict(),
            }
            if tuner._scheduler and hasattr(tuner._scheduler, 'state_dict'):
                state['scheduler'] = tuner._scheduler.state_dict()
            paddle.save(state, path=self._get_file_path(tuner))

        self._saved_checkpoints.append(self._get_file_path(tuner))

    def _get_file_path(self, tuner):
        """
        Returns the file path for checkpoint.
        """
        return os.path.join(
            self._save_dir, f'saved_model_epoch_{tuner.state.epoch + 1:02d}'
        )

    @staticmethod
    def load(tuner: 'BaseTuner', fp: str):
        """
        Loads the model and tuner state
        """
        framework = get_framework(tuner.embed_model)

        if framework == 'keras':
            import keras

            tuner._embed_model = keras.models.load_model(fp)
            with open(os.path.join(fp, 'saved_state.pkl'), 'rb') as f:
                loaded_state = pickle.load(f)
            tuner.state.epoch = loaded_state['epoch']

        elif framework == 'torch':
            import torch

            checkpoint = torch.load(fp)
            tuner._embed_model.load_state_dict(checkpoint['state_dict'])
            tuner._optimizer.load_state_dict(checkpoint['optimizer'])
            if tuner._scheduler and hasattr(tuner._scheduler, 'state_dict'):
                tuner._scheduler.load_state_dict(checkpoint['scheduler'])
            tuner.state.epoch = checkpoint['epoch']

        elif framework == 'paddle':
            import paddle

            checkpoint = paddle.load(fp)
            tuner._embed_model.set_state_dict(checkpoint['state_dict'])
            tuner._optimizer.set_state_dict(checkpoint['optimizer'])
            if tuner._scheduler and hasattr(tuner._scheduler, 'state_dict'):
                tuner._scheduler.set_state_dict(checkpoint['scheduler'])
            tuner.state.epoch = checkpoint['epoch']
