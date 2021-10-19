import numpy as np
import pytest

from finetuner.toydata import generate_fashion_match
from finetuner.tuner.base import BaseDataset
from finetuner.tuner.dataset import SiameseMixin, TripletMixin


@pytest.mark.parametrize(
    'data_src',
    [
        generate_fashion_match(num_pos=10, num_neg=10, num_total=100),
        lambda: generate_fashion_match(num_pos=10, num_neg=10, num_total=100),
    ],
)
def test_siamese_dataset(data_src):
    class SD(SiameseMixin, BaseDataset):
        ...

    sd = SD(data_src)
    for d in sd:
        assert len(d) == 2
        assert len(d[0]) == 2
        assert isinstance(d[0][0], np.ndarray)
        assert isinstance(d[0][1], np.ndarray)
        assert d[1] == 1.0 or d[1] == -1.0
        break


@pytest.mark.parametrize(
    'data_src',
    [
        generate_fashion_match(num_pos=10, num_neg=10, num_total=100),
        lambda: generate_fashion_match(num_pos=10, num_neg=10, num_total=100),
    ],
)
def test_triplet_dataset(data_src):
    class SD(TripletMixin, BaseDataset):
        ...

    sd = SD(data_src)
    for d in sd:
        assert len(d) == 2
        assert len(d[0]) == 3
        assert isinstance(d[0][0], np.ndarray)
        assert isinstance(d[0][1], np.ndarray)
        assert isinstance(d[0][2], np.ndarray)
        assert d[1] == 0.0
        break
