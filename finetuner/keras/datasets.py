from ..base import BaseDataset
from ..dataset import SiameseMixin, TripletMixin


class SiameseDataset(SiameseMixin, BaseDataset):
    ...


class TripletDataset(TripletMixin, BaseDataset):
    ...
