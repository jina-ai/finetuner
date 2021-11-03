import numpy as np
from typing import List, Tuple
from itertools import combinations, groupby, permutations

from ..base import BaseMiner
from ...helper import AnyTensor


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
        return [
            (left[0], right[0], 1) if left[1] == right[1] else (left[0], right[0], -1)
            for left, right in combinations(enumerate(labels), 2)
        ]


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
        labels1 = np.expand_dims(labels, 1)
        labels2 = np.expand_dims(labels, 0)
        matches = (labels1 == labels2).astype(int)
        diffs = matches ^ 1
        np.fill_diagonal(matches, 0)
        triplets = np.expand_dims(matches, 2) * np.expand_dims(diffs, 1)
        idxes_anchor, idxes_pos, idxes_neg = np.where(triplets)
        return list(zip(idxes_anchor, idxes_pos, idxes_neg))


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
            session_pairs = []
            for left, right in combinations(enumerate(session_labels), 2):
                if left[1] in [0, 1] and right[1] in [0, 1]:
                    # 0 represents for anchor, 1 for positive, they form positive pairs.
                    session_pairs.append((left[0], right[0], 1))
                elif left[1] == -1 and right[1] == -1:
                    # if both left and right are negatives, skip.
                    continue
                else:
                    # one anchor or positive, one negative form a negative sample.
                    session_pairs.append((left[0], right[0], -1))
            for left_idx, right_idx, label in session_pairs:
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
        :param labels: labels of each embeddings, consist of session id and label, the labels of
          anchor, positive, negative are 0, 1 and -1.
        :return: triplet of label indices follows the order of anchor, positive and negative.
        """
        assert len(embeddings) == len(labels)
        rv = []
        labels_with_index = [item + (index,) for index, item in enumerate(labels)]
        sorted_labels_with_index = sorted(labels_with_index, key=lambda x: x[0])
        for session, group in groupby(sorted_labels_with_index, lambda x: x[0]):
            _, session_labels, session_indices = zip(*group)
            for left, middle, right in permutations(enumerate(session_labels), 3):
                neg_label_count = [left[1], middle[1], right[1]].count(-1)
                # one and only one label is -1 means all rest greater or equal than 0
                # and last element must be negative
                if neg_label_count == 1 and right[1] == -1:
                    rv.append(
                        (
                            session_indices[left[0]],
                            session_indices[middle[0]],
                            session_indices[right[0]],
                        )
                    )
        return rv
