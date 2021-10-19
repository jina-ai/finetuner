import numpy as np
import pytest

from finetuner.toydata import generate_fashion_match_catalog
from finetuner.tuner.base import BaseDataset
from finetuner.tuner.dataset import SiameseMixin, TripletMixin


@pytest.mark.parametrize(
    'pre_init_generator',
    [True, False],
)
def test_siamese_dataset(pre_init_generator):
    class SD(SiameseMixin, BaseDataset):
        ...

    data, catalog = generate_fashion_match_catalog(
        num_pos=10,
        num_neg=10,
        num_total=100,
        pre_init_generator=pre_init_generator,
    )
    sd = SD(data, catalog)
    for d in sd:
        assert len(d) == 2
        assert len(d[0]) == 2
        assert isinstance(d[0][0], np.ndarray)
        assert isinstance(d[0][1], np.ndarray)
        assert d[1] == 1.0 or d[1] == -1.0
        break


@pytest.mark.parametrize(
    'pre_init_generator',
    [True, False],
)
def test_triplet_dataset(pre_init_generator):
    class SD(TripletMixin, BaseDataset):
        ...

    data, catalog = generate_fashion_match_catalog(
        num_pos=10,
        num_neg=10,
        num_total=100,
        pre_init_generator=pre_init_generator,
    )

    sd = SD(data, catalog)
    for d in sd:
        assert len(d) == 2
        assert len(d[0]) == 3
        assert isinstance(d[0][0], np.ndarray)
        assert isinstance(d[0][1], np.ndarray)
        assert isinstance(d[0][2], np.ndarray)
        assert d[1] == 0.0
        break
