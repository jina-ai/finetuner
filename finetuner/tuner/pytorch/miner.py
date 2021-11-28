from cProfile import label
from multiprocessing.sharedctypes import Value
import numpy as np
from typing import Tuple

import torch
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


class TripletHardMiner(TripletMiner):
    def __init__(self, strategies=["hard"]):
        self.strategies = strategies

    def _update_matches(
        self,
        match_mat: torch.Tensor,
        dist_mat: torch.Tensor,
        semihard_tsh: torch.Tensor = None,
    ):
        _filter = self._get_filter_function(self.pos_strategy)
        per_row_max_vals = _filter(match_mat * dist_mat, semihard_tsh)
        return per_row_max_vals * match_mat

    def _update_diffs(
        self,
        diff_mat: torch.Tensor,
        dist_mat: torch.Tensor,
        semihard_tsh: torch.Tensor = None,
    ):
        return

    def _get_per_row_max(
        self, dist_mat: torch.Tensor, semihard_tsh: torch.Tensor = None
    ):
        if semihard_tsh is not None:
            dist_mat[dist_mat > semihard_tsh] = 0

        non_zero_rows = torch.any(dist_mat != 0, dim=1)
        return torch.max(dist_mat, dim=1, keepdim=True)[0], non_zero_rows

    def _get_per_row_min(
        self, dist_mat: torch.Tensor, semihard_tsh: torch.Tensor = None
    ):
        non_zero_mask = dist_mat > 0

        # Prevent wrong neg. samples from being extracted
        row_max = torch.max(dist_mat, dim=1, keepdim=True)[0]
        dist_mat += (row_max + 1) * torch.logical_not(non_zero_mask)

        if semihard_tsh is not None:
            dist_mat[dist_mat < semihard_tsh] = float("inf")

        return torch.min(dist_mat, dim=1, keepdim=True)[0]

    def _get_filter_function(self, strategy: str):
        if strategy in ["hard", "semihard"]:
            return self._get_per_row_min
        elif strategy in "easy":
            return self._get_per_row_max
        else:
            raise ValueError(
                f'Positive and negative strategies have to be one of '
                '[`easy`, `semihard`, `hard`], but was: {strategy}'
            )

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
        # distances = torch.Tensor(
        #     np.array([[0, 184, 222], [184, 0, 45], [222, 45, 0]])
        # )  # distances[:3, :3]
        # labels = labels[:3]
        np.random.seed(1234)
        distances = torch.Tensor(np.round_(np.random.random((6, 6)), 1))
        distances -= distances * torch.eye(len(distances))
        triplets_super = super().mine(labels, distances)
        assert len(distances) == len(labels)

        labels1, labels2 = labels.unsqueeze(1), labels.unsqueeze(0)
        matches = (labels1 == labels2).byte()
        diffs = matches ^ 1

        matches.fill_diagonal_(0)
        triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)

        # Get all d(a, p)
        d_a_p = matches * distances
        # Get all d(a, n)
        d_a_n = diffs * distances

        triplets = torch.where(triplets)
        triplet_idxs = torch.concat(triplets).view(3, len(triplets[0]))
        # pos_distances = d_a_p[triplets[0], triplets[1]]
        # neg_distances = d_a_n[triplets[0], triplets[2]]

        match_mask = torch.zeros_like(distances).bool()
        diff_mask = torch.zeros_like(distances).bool()

        if "hard" in self.strategies:
            # Get hardest negative samples
            neg_distances = self._get_per_row_min(diffs * distances)
            neg_mask = neg_distances == d_a_n
            diff_mask = torch.logical_or(diff_mask, neg_mask)

            # Get all pos samples with larger distance than neg one
            pos_mask = neg_distances < d_a_p
            match_mask = torch.logical_or(match_mask, pos_mask)

        if "semihard" in self.strategies:
            # Get hardest negative sample
            neg_distances = self._get_per_row_min(diffs * distances)
            neg_mask = neg_distances < d_a_n
            diff_mask = torch.logical_or(diff_mask, neg_mask)

            # Get hardest pos samples that are further than neg samples
            pos_distances = self._get_per_row_min(d_a_p, neg_distances)
            pos_mask = pos_distances == d_a_p
            match_mask = torch.logical_or(match_mask, pos_mask)

        if "easy" in self.strategies:
            # Get easy positive sample
            pos_distances, _ = self._get_per_row_min(matches * distances)
            pos_mask = pos_distances == d_a_p
            match_mask = torch.logical_or(match_mask, pos_mask)

            # Get easy negative samples, further away than pos sample
            neg_mask = pos_distances < d_a_n
            diff_mask = torch.logical_or(diff_mask, neg_mask)

        import pdb

        pdb.set_trace()

        return torch.where(triplets)


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
