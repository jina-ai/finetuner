import numpy as np
from typing import List, Tuple
from itertools import combinations, groupby

from ..base import BaseMiner
from ...helper import AnyTensor


def _generate_all_possible_pairs(labels: List[int]):
    return [
        (left[0], right[0], 1) if left[1] == right[1] else (left[0], right[0], -1)
        for left, right in combinations(enumerate(labels), 2)
    ]


def _generate_all_possible_triplets(labels: List[int]):
    labels1 = np.expand_dims(labels, 1)
    labels2 = np.expand_dims(labels, 0)
    matches = (labels1 == labels2).astype(int)
    diffs = matches ^ 1
    np.fill_diagonal(matches, 0)
    triplets = np.expand_dims(matches, 2) * np.expand_dims(diffs, 1)
    idxes_anchor, idxes_pos, idxes_neg = np.where(triplets)
    return list(zip(idxes_anchor, idxes_pos, idxes_neg))


class SiameseMiner(BaseMiner):
    def mine(
        self, embeddings: AnyTensor, labels: List[int]
    ) -> List[Tuple[int, int, int]]:
        """Generate tuples from input embeddings and labels.

        :param embeddings: embeddings from model, should be a list of Tensor objects.
        :param labels: labels of each embeddings, embeddings with same label indicates same class.
        :return: a pair of label indices and their label as tuple.
        """
        assert len(embeddings) == len(labels)
        return _generate_all_possible_pairs(labels)


class TripletMiner(BaseMiner):
    def mine(
        self, embeddings: AnyTensor, labels: List[int]
    ) -> List[Tuple[int, int, int]]:
        """Generate triplets from input embeddings and labels.

        :param embeddings: embeddings from model, should be a list of Tensor objects.
        :param labels: labels of each embeddings, embeddings with same label indicates same class.
        :return: triplet of label indices follows the order of anchor, positive and negative.
        """
        assert len(embeddings) == len(labels)
        return _generate_all_possible_triplets(labels)


class SiameseSessionMiner(BaseMiner):
    def mine(
        self, embeddings: AnyTensor, labels: List[Tuple[int, int]]
    ) -> List[Tuple[int, int, int]]:
        """Generate tuples from input embeddings and labels.

        :param embeddings: embeddings from model, should be a list of Tensor objects.
        :param labels: labels of each embeddings, consist of session id and label.
          the labels are either -1 or 1.
        :return: a pair of label indices and their label as tuple.
        """
        assert len(embeddings) == len(labels)
        rv = []
        labels_with_index = [item + (index,) for index, item in enumerate(labels)]
        sorted_labels_with_index = sorted(labels_with_index, key=lambda x: x[0])
        for session, group in groupby(sorted_labels_with_index, lambda x: x[0]):
            _, session_labels, session_indices = zip(*group)
            pairs = _generate_all_possible_pairs(session_labels)
            for left_idx, right_idx, label in pairs:
                rv.append(
                    (session_indices[left_idx], session_indices[right_idx], label)
                )
        return rv


class TripletSessionMiner(BaseMiner):
    def mine(
        self, embeddings: AnyTensor, labels: List[Tuple[int, int]]
    ) -> List[Tuple[int, int, int]]:
        """Generate triplets from input embeddings and labels.

        :param embeddings: embeddings from model, should be a list of Tensor objects.
        :param labels: labels of each embeddings, consist of session id and label.
        :return: triplet of label indices follows the order of anchor, positive and negative.
        """
        assert len(embeddings) == len(labels)
        rv = []
        labels_with_index = [item + (index,) for index, item in enumerate(labels)]
        sorted_labels_with_index = sorted(labels_with_index, key=lambda x: x[0])
        for session, group in groupby(sorted_labels_with_index, lambda x: x[0]):
            _, session_labels, session_indices = zip(*group)
            triplets = _generate_all_possible_triplets(session_labels)
            for left_idx, middle_idx, right_idx in triplets:
                rv.append(
                    (
                        session_indices[left_idx],
                        session_indices[middle_idx],
                        session_indices[right_idx],
                    )
                )
        return rv
