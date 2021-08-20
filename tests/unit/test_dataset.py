import numpy as np
import pytest

from tests.data_generator import fashion_match_doc_generator as fmdg
from tests.data_generator import fashion_match_documentarray as fmda
from trainer.dataset import SiameseMixin, Dataset, TripletMixin


@pytest.mark.parametrize(
    'data_src',
    [fmdg(num_total=100), fmda(num_total=100), fmdg, lambda: fmda(num_total=100)],
)
def test_siamese_dataset(data_src):
    class SD(SiameseMixin, Dataset):
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
    [fmdg(num_total=100), fmda(num_total=100), fmdg, lambda: fmda(num_total=100)],
)
def test_triplet_dataset(data_src):
    class SD(TripletMixin, Dataset):
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
