from torch.utils.data import Dataset

from ..dataset import ClassDataset, SessionDataset


class PytorchClassDataset(ClassDataset, Dataset):
    ...


class PytorchSessionDataset(SessionDataset, Dataset):
    ...
