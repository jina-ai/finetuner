from copy import deepcopy
from typing import Tuple, TYPE_CHECKING

import numpy as np
from torch.utils.data import Dataset as PytorchDataset

from ..dataset import ClassDataset, SessionDataset

if TYPE_CHECKING:
    from jina.types.document.mixins.content import DocumentContentType


class PytorchClassDataset(ClassDataset, PytorchDataset):
    def __getitem__(self, ind: int) -> Tuple['DocumentContentType', int]:
        content, label = super().__getitem__(ind)
        if isinstance(content, np.ndarray) and (content.flags['WRITEABLE'] == False):
            return (deepcopy(content), label)
        return (content, label)


class PytorchSessionDataset(SessionDataset, PytorchDataset):
    def __getitem__(self, ind: int) -> Tuple['DocumentContentType', int]:
        content, label = super().__getitem__(ind)
        if isinstance(content, np.ndarray) and (content.flags['WRITEABLE'] == False):
            return (deepcopy(content), label)
        return (content, label)
