from paddle.io import Dataset as PaddleDataset

from ..dataset import ClassDataset, SessionDataset


class PaddleClassDataset(ClassDataset, PaddleDataset):
    ...


class PaddleSessionDataset(SessionDataset, PaddleDataset):
    ...
