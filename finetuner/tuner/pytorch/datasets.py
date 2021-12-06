from copy import deepcopy
from typing import Tuple, TYPE_CHECKING

import numpy as np
from torch.utils.data import Dataset as PytorchDataset

from ..dataset import ClassDataset, SessionDataset

if TYPE_CHECKING:
    from jina.types.document.mixins.content import DocumentContentType


def _make_blob_writable(content):
    if isinstance(content, np.ndarray):
        # jina blob is immutable, see content.flags['WRITEABLE']
        return deepcopy(content)
    return content


class PytorchClassDataset(ClassDataset, PytorchDataset):
    def __getitem__(self, ind: int) -> Tuple['DocumentContentType', int]:
        content, label = super().__getitem__(ind)
        content = _make_blob_writable(content)
        return (content, label)


class PytorchSessionDataset(SessionDataset, PytorchDataset):
    def __getitem__(self, ind: int) -> Tuple['DocumentContentType', Tuple[int, int]]:
        content, label = super().__getitem__(ind)
        content = _make_blob_writable(content)
        return (content, label)
