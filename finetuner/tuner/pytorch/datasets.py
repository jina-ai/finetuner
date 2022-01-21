from torch.utils.data import Dataset as PytorchDataset

from ..dataset import ClassDataset, InstanceDataset, SessionDataset


class PytorchClassDataset(ClassDataset, PytorchDataset):
    pass


class PytorchSessionDataset(SessionDataset, PytorchDataset):
    pass


class PytorchInstanceDataset(InstanceDataset, PytorchDataset):
    pass
