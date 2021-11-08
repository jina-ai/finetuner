from itertools import combinations, groupby, permutations
from typing import Tuple

import torch

from ..base import BaseClassMiner, BaseSessionMiner


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
        ind_one, ind_two, labels_ret = [], [], []

        labels_index = [
            (sess, match_type, index)
            for index, (sess, match_type) in enumerate(zip(sessions, match_types))
        ]
        sorted_labels_with_index = sorted(labels_index, key=lambda x: x[0])

        for _, group in groupby(sorted_labels_with_index, lambda x: x[0]):
            for left, right in combinations(group, 2):  # (session_id, label, ind)
                if left[1] != -1 or right[1] != -1:
                    ind_one.append(left[2])
                    ind_two.append(right[2])
                    labels_ret.append(0 if min(left[1], right[1]) == -1 else 1)

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
        anchor_ind, pos_ind, neg_ind = [], [], []

        labels_index = [
            (sess, match_type, index)
            for index, (sess, match_type) in enumerate(zip(sessions, match_types))
        ]
        sorted_labels_with_index = sorted(labels_index, key=lambda x: x[0])

        for _, group in groupby(sorted_labels_with_index, lambda x: x[0]):
            anchor_pos_session_indices = []
            neg_session_indices = []

            for _, session_label, session_index in group:
                if session_label >= 0:
                    anchor_pos_session_indices.append(session_index)
                else:
                    neg_session_indices.append(session_index)

            for anchor, pos in permutations(anchor_pos_session_indices, 2):
                anchor_ind += [anchor] * len(neg_session_indices)
                pos_ind += [pos] * len(neg_session_indices)
                neg_ind += neg_session_indices

        return (
            torch.tensor(anchor_ind, device=distances.device),
            torch.tensor(pos_ind, device=distances.device),
            torch.tensor(neg_ind, device=distances.device),
        )
