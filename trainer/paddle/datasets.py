from paddle.io import IterableDataset

from ..base import BaseDataset
from ..dataset import SiameseMixin, TripletMixin


class SiameseDataset(SiameseMixin, BaseDataset, IterableDataset):
    ...


class TripletDataset(TripletMixin, BaseDataset, IterableDataset):
    ...


def get_dataset(arity):
    if arity == 2:

        return SiameseDataset
    elif arity == 3:

        return TripletDataset
    else:
        raise NotImplementedError
