from typing import List, Mapping, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from ..dataset import ClassDataset, SessionDataset
    from ...helper import CollateFnType


def _default_collate(content: List):
    if isinstance(content[0], np.ndarray):
        return np.array(content)
    elif isinstance(content[0], Sequence):
        return (np.array([x[i] for x in content]) for i in range(len(content[0])))
    elif isinstance(content[0], Mapping):
        return {k: np.array([x[k] for x in content]) for k in content[0].keys()}


class KerasDataSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        batch_sampler,
        dataset: Union['ClassDataset', 'SessionDataset'],
        collate_fn: Optional['CollateFnType'] = None,
    ):
        self.dataset = dataset
        self.batch_sampler = batch_sampler

        self.batches = list(iter(self.batch_sampler))
        self.collate_fn = collate_fn or _default_collate

    def __getitem__(self, idx: int):
        """Get batch"""
        batch_ids = self.batches[idx]
        items = [self.dataset[bidx] for bidx in batch_ids]

        content = self.collate_fn([x[0] for x in items])

        if isinstance(items[0][1], int):
            labels = np.array([x[1] for x in items])
        else:
            labels = (
                np.array([x[1][0] for x in items]),
                np.array([x[1][1] for x in items]),
            )

        return content, labels

    def __len__(self):
        return len(self.batches)

    def on_epoch_end(self):
        """Re-create batches at the end of each epoch"""
        self.batches = list(iter(self.batch_sampler))
