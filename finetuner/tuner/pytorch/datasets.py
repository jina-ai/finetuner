from torch.utils.data import Dataset as PytorchDataset

from ..dataset import ClassDataset, SessionDataset


class PytorchClassDataset(ClassDataset, PytorchDataset):
    ...


class PytorchSessionDataset(SessionDataset, PytorchDataset):
    ...
