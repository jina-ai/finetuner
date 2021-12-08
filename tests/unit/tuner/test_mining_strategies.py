import numpy as np
import pytest
import torch

from finetuner.tuner.miner.mining_strategies import TorchStrategicMiningHelper


@pytest.fixture()
def dummy_distances():
    return torch.Tensor(
        (
            (0, 4, 3, 7, 7, 6),
            (4, 0, 2, 5, 7, 7),
            (3, 2, 0, 5, 6, 6),
            (7, 5, 5, 0, 3, 5),
            (7, 7, 6, 3, 0, 3),
            (6, 7, 6, 5, 3, 0),
        )
    )


@pytest.fixture
def labels():
    return torch.tensor([1, 3, 1, 3, 2, 2])


def test_torch_strategic_mining_helper_inputs():

    with pytest.raises(ValueError) as err:
        TorchStrategicMiningHelper(pos_strategy='some wrong', neg_strategy='arguments')
    with pytest.raises(ValueError) as err:
        TorchStrategicMiningHelper(pos_strategy='semihard', neg_strategy='all')
    with pytest.raises(ValueError) as err:
        TorchStrategicMiningHelper(pos_strategy='all', neg_strategy='semihard')

    with pytest.raises(ValueError) as err:
        TorchStrategicMiningHelper(pos_strategy='semihard', neg_strategy='semihard')


def test_torch_strategic_mining_helper_neg_neg(labels, dummy_distances):

    labels1, labels2 = labels.unsqueeze(1), labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1

    matches.fill_diagonal_(0)

    strategic_mining_helper = TorchStrategicMiningHelper(
        pos_strategy='all', neg_strategy='all'
    )

    unchanged_matches, unchanged_diffs = strategic_mining_helper.apply_strategy(
        matches, diffs, dummy_distances
    )

    np.testing.assert_equal(matches, unchanged_matches.numpy())
    np.testing.assert_equal(diffs, unchanged_diffs.numpy())


def test_torch_strategic_mining_semihard_thresholding_row_max(labels):

    distances = torch.Tensor(((0, 1), (1, 0)))
    semihard_tsh = torch.Tensor((2, 0.5)).unsqueeze(1)

    strategic_mining_helper = TorchStrategicMiningHelper('hard', 'semihard')
    tmp = strategic_mining_helper._get_per_row_max(distances, semihard_tsh)
    (_, _), invalid_row_mask = tmp

    distances[invalid_row_mask] = 0
    np.testing.assert_array_equal(distances, torch.Tensor(((0, 1), (0, 0))))


def test_torch_strategic_mining_semihard_thresholding_row_min(labels):

    distances = torch.Tensor(((0, 1), (1, 0)))
    semihard_tsh = torch.Tensor((2, 0.5)).unsqueeze(1)

    strategic_mining_helper = TorchStrategicMiningHelper('semihard', 'hard')
    tmp = strategic_mining_helper._get_per_row_min(torch.clone(distances), semihard_tsh)
    (_, _), invalid_row_mask = tmp

    distances[invalid_row_mask] = 0
    np.testing.assert_array_equal(distances, torch.Tensor(((0, 0), (1, 0))))
