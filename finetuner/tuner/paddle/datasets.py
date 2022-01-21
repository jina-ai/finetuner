from paddle.io import Dataset as PaddleDataset

from ..dataset import ClassDataset, InstanceDataset, SessionDataset


class PaddleClassDataset(ClassDataset, PaddleDataset):
    ...


class PaddleSessionDataset(SessionDataset, PaddleDataset):
    ...


class PaddleInstanceDataset(InstanceDataset, PaddleDataset):
    ...
