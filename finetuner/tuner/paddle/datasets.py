from paddle.io import IterableDataset

from ..base import BaseDataset
from ..dataset import SiameseMixin, TripletMixin


class SiameseDataset(SiameseMixin, BaseDataset, IterableDataset):
    ...


class TripletDataset(TripletMixin, BaseDataset, IterableDataset):
    ...
