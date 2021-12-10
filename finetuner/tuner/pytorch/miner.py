from typing import Tuple

import torch

from ..miner import get_session_pairs, get_session_triplets
from ..miner.base import BaseClassMiner, BaseSessionMiner
from ..miner.mining_strategies import TorchStrategicMiningHelper


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


class SiameseEasyHardMiner(BaseClassMiner[torch.Tensor]):
    def __init__(self, pos_strategy: str = 'hard', neg_strategy: str = 'hard'):
        """
        Miner implements easy-hard mining for tuples in siamese training.
        The following strategies are available.

        Pos. Strategy:
        - 'hard': Returns hardest positive (furthest) sample per anchor
        - 'semihard': Returns the hardest positive sample per anchor, such
          that it is closer than the selected negative
        - 'easy': Returns the easiest positive sample per anchor
        - 'all': Returns all positive samples

        Neg. Strategy:
        - 'hard': Returns hardest negative (closest) sample per anchor
        - 'semihard': Returns the hardest negative sample per anchor, such
          that it is further than the selected negative
        - 'easy': Returns the easiest negative sample per anchor
        - 'all': Returns all negative samples

        Not allowed:
        - pos. and neg. strategy cannot be set to 'semihard' simultaneously
        - When pos. or neg. strategy is set to 'semihard' the other cannot be
          set to 'all'

        :param pos_strategy: Strategy for selecting positive samples
        :param neg_strategy: Strategy for selecting negative samples
        """

        self.strategic_mining_helper = TorchStrategicMiningHelper(
            pos_strategy, neg_strategy
        )

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

        matches, diffs = self.strategic_mining_helper.apply_strategy(
            matches, diffs, distances
        )

        ind1_pos, ind2_pos = torch.where(matches)
        ind1_neg, ind2_neg = torch.where(diffs)

        ind1 = torch.cat([ind1_pos, ind1_neg])
        ind2 = torch.cat([ind2_pos, ind2_neg])
        target = torch.cat([torch.ones_like(ind1_pos), torch.zeros_like(ind1_neg)])

        return ind1, ind2, target


class TripletEasyHardMiner(BaseClassMiner[torch.Tensor]):
    def __init__(self, pos_strategy='hard', neg_strategy='hard'):
        """
        Miner implements easy-hard mining for triplets during training with
        triplet loss. The following strategies are available.

        Pos. Strategy:
        - 'hard': Returns hardest positive (furthest) sample per anchor
        - 'semihard': Returns the hardest positive sample per anchor, such
          that it is closer than the selected negative
        - 'easy': Returns the easiest positive sample per anchor
        - 'all': Returns all positive samples

        Neg. Strategy:
        - 'hard': Returns hardest negative (closest) sample per anchor
        - 'semihard': Returns the hardest negative sample per anchor, such
          that it is further than the selected negative
        - 'easy': Returns the easiest negative sample per anchor
        - 'all': Returns all negative samples

        Not allowed:
        - pos. and neg. strategy cannot be set to 'semihard' simultaneously
        - When pos. or neg. strategy is set to 'semihard' the other cannot be
          set to 'all'

        :param pos_strategy: Strategy for selecting positive samples
        :param neg_strategy: Strategy for selecting negative samples
        """
        self.strategic_mining_helper = TorchStrategicMiningHelper(
            pos_strategy, neg_strategy
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
        assert len(distances) == len(labels)

        labels1, labels2 = labels.unsqueeze(1), labels.unsqueeze(0)
        matches = (labels1 == labels2).byte()
        diffs = matches ^ 1

        matches.fill_diagonal_(0)
        matches, diffs = self.strategic_mining_helper.apply_strategy(
            matches, diffs, distances
        )

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
