import abc
from itertools import combinations, groupby, permutations
from typing import Generic, List, Sequence, Tuple, TypeVar

import numpy as np


LabelType = TypeVar('LabelType')


class BaseMiner(abc.ABC, Generic[LabelType]):
    @abc.abstractmethod
    def mine(
        self, labels: Sequence[LabelType], distances: np.ndarray
    ) -> List[Tuple[int, ...]]:
        """Generate tuples/triplets from input embeddings and labels.

        :param labels: labels of each embeddings
        :param distances: A matrix of pairwise distance between each two item embeddings

        :return: tuple/triplet of label indices.
        """
        ...


class SiameseMiner(BaseMiner[int]):
    def mine(
        self, labels: Sequence[int], distances: np.ndarray
    ) -> List[Tuple[int, ...]]:
        """Generate tuples from input embeddings and labels.

        :param labels: labels of each embeddings, embeddings with same label indicates
            same class.
        :param distances: A matrix of pairwise distance between each two item embeddings

        :return: a pair of label indices and their label as tuple.
        """
        assert len(distances) == len(labels)
        return [
            (left[0], right[0], 1) if left[1] == right[1] else (left[0], right[0], 0)
            for left, right in combinations(enumerate(labels), 2)
        ]


class TripletMiner(BaseMiner[int]):
    def mine(
        self, labels: Sequence[int], distances: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Generate triplets from input embeddings and labels.

        :param labels: labels of each embeddings, embeddings with same label indicates
            same class.
        :param distances: A matrix of pairwise distance between each two item embeddings

        :return: triplet of label indices follows the order of anchor, positive and
            negative.
        """
        assert len(distances) == len(labels)

        labels1 = np.expand_dims(labels, 1)
        labels2 = np.expand_dims(labels, 0)

        matches = (labels1 == labels2).astype(int)
        diffs = matches ^ 1
        np.fill_diagonal(matches, 0)

        triplets = np.expand_dims(matches, 2) * np.expand_dims(diffs, 1)
        idxes_anchor, idxes_pos, idxes_neg = np.where(triplets)
        return list(zip(idxes_anchor, idxes_pos, idxes_neg))


class SiameseSessionMiner(BaseMiner[Tuple[int, int]]):
    def mine(
        self, labels: Sequence[Tuple[int, int]], distances: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Generate tuples from input embeddings and labels.

        :param labels: labels of each embeddings, consist of session id and label.
          the labels are either -1 or 1.
        :param distances: A matrix of pairwise distance between each two item embeddings

        :return: a pair of label indices and their label as tuple.
        """
        assert len(distances) == len(labels)

        rv = []

        labels_with_index = [item + (index,) for index, item in enumerate(labels)]
        sorted_labels_with_index = sorted(labels_with_index, key=lambda x: x[0])

        for _, group in groupby(sorted_labels_with_index, lambda x: x[0]):
            _, session_labels, session_indices = zip(*group)
            for left, right in combinations(enumerate(session_labels), 2):

                # Both positive or positive-anchor
                if left[1] >= 0 and right[1] >= 0:
                    rv.append((session_indices[left[0]], session_indices[right[0]], 1))

                # Both negative, relatinship not known
                elif left[1] == -1 and right[1] == -1:
                    pass

                # one of the item is positive or anchor, another one is negative
                else:
                    rv.append((session_indices[left[0]], session_indices[right[0]], 0))
        return rv


class TripletSessionMiner(BaseMiner[Tuple[int, int]]):
    def mine(
        self, labels: Sequence[Tuple[int, int]], distances: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Generate triplets from input embeddings and labels.

        :param labels: labels of each embeddings, consist of session id and label,
            the labels of anchor, positive, negative are 0, 1 and -1.
        :param distances: A matrix of pairwise distance between each two item embeddings

        :return: triplet of label indices follows the order of anchor, positive and
            negative.
        """
        assert len(distances) == len(labels)

        rv = []

        labels_with_index = [item + (index,) for index, item in enumerate(labels)]
        sorted_labels_with_index = sorted(labels_with_index, key=lambda x: x[0])

        for _, group in groupby(sorted_labels_with_index, lambda x: x[0]):
            anchor_pos_session_labels = []
            anchor_pos_session_indices = []
            neg_session_indices = []

            for _, session_label, session_index in group:
                if session_label >= 0:
                    anchor_pos_session_labels.append(session_label)
                    anchor_pos_session_indices.append(session_index)
                else:
                    neg_session_indices.append(session_index)

            for anchor, pos in permutations(enumerate(anchor_pos_session_labels), 2):
                for neg_idx in neg_session_indices:
                    rv.append(
                        (
                            anchor_pos_session_indices[anchor[0]],
                            anchor_pos_session_indices[pos[0]],
                            neg_idx,
                        )
                    )
        return rv
