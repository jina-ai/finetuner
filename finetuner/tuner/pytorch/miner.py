import numpy as np
from typing import Optional, Tuple
from numpy.lib.function_base import diff

import torch
from torch._C import Value
from torch.nn.modules import distance

from ..miner import get_session_pairs, get_session_triplets
from ..miner.base import BaseClassMiner, BaseSessionMiner


class SiameseMiner(BaseClassMiner[torch.Tensor]):
    def mine(
        self, labels: torch.Tensor, distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all possible pairs.

        :param labels: A 1D tensor of item labels (classes)
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three 1D tensors, first one holding integers of first element of
            pair, second of the second element of pair, and third one the label (0 or
            1) for the pair for each pair
        """
        assert len(distances) == len(labels)

        l1, l2 = labels.unsqueeze(1), labels.unsqueeze(0)

        matches = (l1 == l2).byte()
        diffs = matches ^ 1
        matches.triu_(1)
        diffs.triu_()

        ind1_pos, ind2_pos = torch.where(matches)
        ind1_neg, ind2_neg = torch.where(diffs)

        ind1 = torch.cat([ind1_pos, ind1_neg])
        ind2 = torch.cat([ind2_pos, ind2_neg])
        target = torch.cat([torch.ones_like(ind1_pos), torch.zeros_like(ind1_neg)])

        return ind1, ind2, target


class TripletMiner(BaseClassMiner[torch.Tensor]):
    def mine(
        self, labels: torch.Tensor, distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all possible triplets.

        :param labels: A 1D tensor of item labels (classes)
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three 1D tensors, holding the anchor index, positive index and
            negative index of each triplet, respectively
        """
        assert len(distances) == len(labels)

        labels1, labels2 = labels.unsqueeze(1), labels.unsqueeze(0)
        matches = (labels1 == labels2).byte()
        diffs = matches ^ 1

        matches.fill_diagonal_(0)

        triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)

        return torch.where(triplets)


def apply_strategy(pos_strategy, neg_strategy, match_mat, diff_mat, dist_mat):

    allowed_strategies = ['easy', 'semihard', 'hard', 'all']
    if pos_strategy not in allowed_strategies or neg_strategy not in allowed_strategies:
        raise ValueError(
            f'The strategy has to be one of all, easy, semihard, and hard, but '
            'was: {strategy}'
        )
    elif pos_strategy == 'semihard' and neg_strategy == 'semihard':
        raise ValueError(
            'Positive and negative strategy cannot both be set to semihard.'
        )
    elif (pos_strategy == 'all' and neg_strategy == 'semihard') or (
        pos_strategy == 'semihard' and neg_strategy == 'all'
    ):
        raise ValueError(
            'When one strategy is set to semihard, the other cannot be set to hard.'
        )

    def _get_per_row_min(
        dist_mat: torch.Tensor, semihard_tsh: Optional[torch.Tensor] = None
    ):
        zero_element_mask = torch.logical_not(dist_mat > 0)

        if len(dist_mat) == 0:
            return dist_mat, torch.empty((), dtype=torch.bool)

        # Set zeros to max value, so they are not extracted as row minimum
        row_max = torch.max(dist_mat, dim=1, keepdim=True)[0]
        dist_mat += (row_max + 1) * zero_element_mask

        if semihard_tsh is not None:
            dist_mat[dist_mat < semihard_tsh] = float('inf')

        non_inf_rows = torch.all(dist_mat == float('inf'), dim=1)
        return torch.min(dist_mat, dim=1, keepdim=True), non_inf_rows

    def _get_per_row_max(
        dist_mat: torch.Tensor, semihard_tsh: Optional[torch.Tensor] = None
    ):

        if len(dist_mat) == 0:
            return dist_mat, torch.empty((), dtype=torch.bool)

        if semihard_tsh is not None:
            dist_mat[dist_mat > semihard_tsh] = 0

        non_zero_rows = torch.all(dist_mat != 0, dim=1)
        return torch.max(dist_mat, dim=1, keepdim=True), non_zero_rows

    def _update_dist_mat(dist_mat, indices):
        keep_mask = torch.zeros_like(dist_mat)
        keep_mask[range(keep_mask.shape[0]), indices.squeeze()] = 1
        # Mask and return the distance matrix
        return dist_mat * keep_mask

    def _get_mine_func(strategy: str):
        if strategy in ['hard', 'semihard']:
            return _get_per_row_max
        else:
            return _get_per_row_min

    def update_pos_mat(match_mat, dist_mat, pos_strategy, semihard_tsh=None):

        # Get all d(a, p)
        d_a_p = match_mat * dist_mat

        mine_func = _get_mine_func(pos_strategy)
        (pos_dists, min_max_indices), invalid_row_mask = mine_func(d_a_p, semihard_tsh)
        match_mat[invalid_row_mask] = 0
        return _update_dist_mat(match_mat, min_max_indices), pos_dists

    def update_neg_mat(diff_mat, dist_mat, neg_strategy, semihard_tsh=None):
        # Get all d(a, n)
        d_a_n = diff_mat * dist_mat

        # Neg. needs to be handled in opposite fashion than pos. strategy
        print(neg_strategy, "kjsdhflksjdhfkldjhsf")
        neg_strategy = 'easy' if neg_strategy in ('hard', 'semihard') else 'hard'
        mine_func = _get_mine_func(neg_strategy)
        (neg_dists, min_max_indices), invalid_row_mask = mine_func(d_a_n, semihard_tsh)
        diff_mat[invalid_row_mask] = 0
        return _update_dist_mat(diff_mat, min_max_indices), neg_dists

    # Impossible combinations:
    # both cannot be semihard
    # If one is semihard, the other cannot be all

    # pos:
    # Hard -> Easy, hard, all
    # Semihard -> get next furthest neg
    # Easy -> Easy, hard, all
    if pos_strategy == 'semihard' and neg_strategy != 'all':

        diff_mat, neg_dists = update_neg_mat(diff_mat, dist_mat, neg_strategy)
        match_mat, _ = update_pos_mat(match_mat, dist_mat, pos_strategy, neg_dists)
    elif pos_strategy != 'all' and neg_strategy == 'semihard':

        match_mat, pos_dists = update_pos_mat(match_mat, dist_mat, pos_strategy)
        diff_mat, _ = update_neg_mat(diff_mat, dist_mat, neg_strategy, pos_dists)
    else:
        if pos_strategy != 'all':
            match_mat, _ = update_pos_mat(match_mat, dist_mat, pos_strategy)
        if neg_strategy != 'all':
            diff_mat, _ = update_neg_mat(diff_mat, dist_mat, neg_strategy)

    return match_mat, diff_mat


class TripletEasyHardMiner(BaseClassMiner[torch.Tensor]):
    def __init__(self, pos_strategy='hard', neg_strategy='hard'):
        self.pos_strategy = pos_strategy
        self.neg_strategy = neg_strategy

    def mine(
        self, labels: torch.Tensor, distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all possible triplets.

        :param labels: A 1D tensor of item labels (classes)
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three 1D tensors, holding the anchor index, positive index and
            negative index of each triplet, respectively
        """
        assert len(distances) == len(labels)

        labels1, labels2 = labels.unsqueeze(1), labels.unsqueeze(0)
        matches = (labels1 == labels2).byte()
        diffs = matches ^ 1

        matches.fill_diagonal_(0)
        matches, diffs = apply_strategy(
            self.pos_strategy, self.neg_strategy, matches, diffs, distances
        )
        triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)

        return torch.where(triplets)


class SiameseEasyHardMiner(BaseClassMiner[torch.Tensor]):
    def __init__(self, pos_strategy: str = 'hard', neg_strategy: str = 'hard'):
        self.pos_strategy = pos_strategy
        self.neg_strategy = neg_strategy

    def mine(
        self, labels: torch.Tensor, distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all possible pairs.

        :param labels: A 1D tensor of item labels (classes)
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three 1D tensors, first one holding integers of first element of
            pair, second of the second element of pair, and third one the label (0 or
            1) for the pair for each pair
        """
        assert len(distances) == len(labels)

        l1, l2 = labels.unsqueeze(1), labels.unsqueeze(0)

        matches = (l1 == l2).byte()
        diffs = matches ^ 1
        matches.triu_(1)
        diffs.triu_()

        matches, diffs = apply_strategy(
            self.pos_strategy, self.neg_strategy, matches, diffs, distances
        )

        ind1_pos, ind2_pos = torch.where(matches)
        ind1_neg, ind2_neg = torch.where(diffs)

        ind1 = torch.cat([ind1_pos, ind1_neg])
        ind2 = torch.cat([ind2_pos, ind2_neg])
        target = torch.cat([torch.ones_like(ind1_pos), torch.zeros_like(ind1_neg)])

        return ind1, ind2, target


class SiameseSessionMiner(BaseSessionMiner[torch.Tensor]):
    def mine(
        self, labels: Tuple[torch.Tensor, torch.Tensor], distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all possible pairs for each session.

        :param labels: A tuple of 1D tensors, denotind the items' session and match
            type (0 for anchor, 1 for postive match and -1 for negative match),
            respectively
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three numpy arrays, first one holding integers of first element of
            pair, second of the second element of pair, and third one the label (0 or
            1) for the pair for each pair
        """
        assert len(distances) == len(labels[0]) == len(labels[1])

        sessions, match_types = [x.tolist() for x in labels]
        ind_one, ind_two, labels_ret = get_session_pairs(sessions, match_types)

        return (
            torch.tensor(ind_one, device=distances.device),
            torch.tensor(ind_two, device=distances.device),
            torch.tensor(labels_ret, device=distances.device),
        )


class TripletSessionMiner(BaseSessionMiner[torch.Tensor]):
    def mine(
        self, labels: Tuple[torch.Tensor, torch.Tensor], distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all possible triplets for each session.

        :param labels: A tuple of 1D tensors, denotind the items' session and match
            type (0 for anchor, 1 for postive match and -1 for negative match),
            respectively
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three numpy arrays, holding the anchor index, positive index and
            negative index of each triplet, respectively
        """

        assert len(distances) == len(labels[0]) == len(labels[1])

        sessions, match_types = [x.tolist() for x in labels]
        anchor_ind, pos_ind, neg_ind = get_session_triplets(sessions, match_types)

        return (
            torch.tensor(anchor_ind, device=distances.device),
            torch.tensor(pos_ind, device=distances.device),
            torch.tensor(neg_ind, device=distances.device),
        )
