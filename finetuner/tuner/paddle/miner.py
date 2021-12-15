import paddle
import torch

from typing import Tuple

from finetuner.tuner.miner.mining_strategies import TorchStrategicMiningHelper
from ..miner import get_session_pairs, get_session_triplets
from ..miner.base import BaseClassMiner, BaseSessionMiner


def _empty_tensor(dtype: str = 'int64') -> paddle.Tensor:
    return paddle.to_tensor([], dtype=dtype)


class SiameseMiner(BaseClassMiner[paddle.Tensor]):
    def mine(
        self, labels: paddle.Tensor, distances: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Generate all possible pairs.

        :param labels: A 1D tensor of item labels (classes)
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three 1D tensors, first one holding integers of first element of
            pair, second of the second element of pair, and third one the label (0 or
            1) for the pair for each pair
        """
        assert len(distances) == len(labels)

        # Needed, else sigfault if empty
        if labels.size == 0:
            return _empty_tensor(), _empty_tensor(), _empty_tensor()

        l1, l2 = labels.unsqueeze(1), labels.unsqueeze(0)
        matches = paddle.cast(l1 == l2, 'int32')
        diffs = 1 - matches
        matches = paddle.triu(matches, 1)
        diffs = paddle.triu(diffs)

        pos_inds = paddle.nonzero(matches).transpose([1, 0])
        neg_inds = paddle.nonzero(diffs).transpose([1, 0])

        # Checking needed, otherwise segfault if empty
        if pos_inds.size:
            ind1_pos, ind2_pos = paddle.nonzero(matches).transpose([1, 0])
        else:
            ind1_pos, ind2_pos = _empty_tensor(), _empty_tensor()

        if neg_inds.size:
            ind1_neg, ind2_neg = paddle.nonzero(diffs).transpose([1, 0])
        else:
            ind1_neg, ind2_neg = _empty_tensor(), _empty_tensor()

        # Checking needed otherwise error on concat
        if ind1_pos.size + ind1_neg.size:
            ind1 = paddle.concat([ind1_pos, ind1_neg])
            ind2 = paddle.concat([ind2_pos, ind2_neg])
            target = paddle.concat(
                [paddle.ones_like(ind1_pos), paddle.zeros_like(ind1_neg)]
            )
        else:
            ind1, ind2, target = _empty_tensor(), _empty_tensor(), _empty_tensor()

        return ind1, ind2, target


class SiameseEasyHardMiner(BaseClassMiner[paddle.Tensor]):
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
        self, labels: paddle.Tensor, distances: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Generate all possible pairs.

        :param labels: A 1D tensor of item labels (classes)
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three 1D tensors, first one holding integers of first element of
            pair, second of the second element of pair, and third one the label (0 or
            1) for the pair for each pair
        """
        assert len(distances) == len(labels)

        # Needed, else sigfault if empty
        if labels.size == 0:
            return _empty_tensor(), _empty_tensor(), _empty_tensor()

        l1, l2 = labels.unsqueeze(1), labels.unsqueeze(0)
        matches = paddle.cast(l1 == l2, 'int32')
        diffs = 1 - matches
        matches = paddle.triu(matches, 1)
        diffs = paddle.triu(diffs)

        # Apply mining strategy
        updated_matches, updated_diffs = self.strategic_mining_helper.apply_strategy(
            matches.numpy(),
            diffs.numpy(),
            distances.numpy(),
            to_numpy=True,
        )
        matches = paddle.to_tensor(updated_matches, place=matches.place)
        diffs = paddle.to_tensor(updated_diffs, place=diffs.place)

        pos_inds = paddle.nonzero(matches).transpose([1, 0])
        neg_inds = paddle.nonzero(diffs).transpose([1, 0])

        # Checking needed, otherwise segfault if empty
        if pos_inds.size:
            ind1_pos, ind2_pos = paddle.nonzero(matches).transpose([1, 0])
        else:
            ind1_pos, ind2_pos = _empty_tensor(), _empty_tensor()

        if neg_inds.size:
            ind1_neg, ind2_neg = paddle.nonzero(diffs).transpose([1, 0])
        else:
            ind1_neg, ind2_neg = _empty_tensor(), _empty_tensor()

        # Checking needed otherwise error on concat
        if ind1_pos.size + ind1_neg.size:
            ind1 = paddle.concat([ind1_pos, ind1_neg])
            ind2 = paddle.concat([ind2_pos, ind2_neg])
            target = paddle.concat(
                [paddle.ones_like(ind1_pos), paddle.zeros_like(ind1_neg)]
            )
        else:
            ind1, ind2, target = _empty_tensor(), _empty_tensor(), _empty_tensor()

        return ind1, ind2, target


class TripletMiner(BaseClassMiner[paddle.Tensor]):
    def mine(
        self, labels: paddle.Tensor, distances: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Generate all possible triplets.

        :param labels: A 1D tensor of item labels (classes)
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three 1D tensors, holding the anchor index, positive index and
            negative index of each triplet, respectively
        """
        assert len(distances) == len(labels)

        # Needed, else sigfault if empty
        if labels.size == 0:
            return _empty_tensor(), _empty_tensor(), _empty_tensor()

        labels1, labels2 = labels.unsqueeze(1), labels.unsqueeze(0)
        matches = paddle.cast(labels1 == labels2, 'int32')
        diffs = 1 - matches

        matches = paddle.tril(matches, -1) + paddle.triu(matches, 1)
        triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
        triplet_inds = paddle.nonzero(triplets).transpose([1, 0])

        # Checking needed, otherwise segfault if empty
        if triplet_inds.size:
            return triplet_inds
        else:
            return _empty_tensor(), _empty_tensor(), _empty_tensor()


class TripletEasyHardMiner(BaseClassMiner[paddle.Tensor]):
    def __init__(self, pos_strategy: str = 'hard', neg_strategy: str = 'hard'):
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
        self, labels: paddle.Tensor, distances: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Generate all possible triplets.

        :param labels: A 1D tensor of item labels (classes)
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three 1D tensors, holding the anchor index, positive index and
            negative index of each triplet, respectively
        """
        assert len(distances) == len(labels)

        # Needed, else sigfault if empty
        if labels.size == 0:
            return _empty_tensor(), _empty_tensor(), _empty_tensor()

        labels1, labels2 = labels.unsqueeze(1), labels.unsqueeze(0)
        matches = paddle.cast(labels1 == labels2, 'int32')
        diffs = 1 - matches

        # Apply mining strategy
        updated_matches, updated_diffs = self.strategic_mining_helper.apply_strategy(
            torch.Tensor(matches.numpy()),
            torch.Tensor(diffs.numpy()),
            torch.Tensor(distances.numpy()),
            to_numpy=True,
        )
        matches = paddle.to_tensor(updated_matches, place=matches.place)
        diffs = paddle.to_tensor(updated_diffs, place=diffs.place)

        matches = paddle.tril(matches, -1) + paddle.triu(matches, 1)
        triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
        triplet_inds = paddle.nonzero(triplets).transpose([1, 0])

        # Checking needed, otherwise segfault if empty
        if triplet_inds.size:
            return triplet_inds
        else:
            return _empty_tensor(), _empty_tensor(), _empty_tensor()


class SiameseSessionMiner(BaseSessionMiner[paddle.Tensor]):
    def mine(
        self, labels: Tuple[paddle.Tensor, paddle.Tensor], distances: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
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
            paddle.to_tensor(ind_one, place=distances.place),
            paddle.to_tensor(ind_two, place=distances.place),
            paddle.to_tensor(labels_ret, place=distances.place),
        )


class TripletSessionMiner(BaseSessionMiner[paddle.Tensor]):
    def mine(
        self, labels: Tuple[paddle.Tensor, paddle.Tensor], distances: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
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
            paddle.to_tensor(anchor_ind, place=distances.place),
            paddle.to_tensor(pos_ind, place=distances.place),
            paddle.to_tensor(neg_ind, place=distances.place),
        )
