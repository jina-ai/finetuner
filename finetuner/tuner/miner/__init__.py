import abc
from itertools import combinations, groupby, permutations
from typing import Generic, Sequence, Tuple, TypeVar

import numpy as np


LabelType = TypeVar('LabelType')


class BaseMiner(abc.ABC, Generic[LabelType]):
    @abc.abstractmethod
    def mine(
        self, labels: Sequence[LabelType], distances: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Generate tuples/triplets from input embeddings and labels.

        :param labels: labels of each embeddings
        :param distances: A matrix of pairwise distance between each two item embeddings

        :return: A list of numpy arrays, each holding an element of a tuple for
            every tuple
        """
        ...


class SiameseMiner(BaseMiner[int]):
    def mine(
        self, labels: Sequence[int], distances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate tuples from input embeddings and labels.

        :param labels: labels of each embeddings, embeddings with same label indicates
            same class.
        :param distances: A matrix of pairwise distance between each two item embeddings

        :return: three numpy arrays, first one holding integers of first element of
            pair, second of the second element of pair, and third one the label (0 or
            1) for the pair for each pair
        """
        assert len(distances) == len(labels)

        n_comb = (len(labels) * (len(labels) - 1)) // 2
        ind_one = np.empty(n_comb, dtype=np.int32)
        ind_two = np.empty(n_comb, dtype=np.int32)
        labels_ret = np.empty(n_comb, dtype=np.int32)

        for i, (left, right) in enumerate(combinations(enumerate(labels), 2)):
            ind_one[i] = left[0]
            ind_two[i] = right[0]
            labels_ret[i] = 1 if left[1] == right[1] else 0

        return ind_one, ind_two, labels_ret


class TripletMiner(BaseMiner[int]):
    def mine(
        self, labels: Sequence[int], distances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate triplets from input embeddings and labels.

        :param labels: labels of each embeddings, embeddings with same label indicates
            same class.
        :param distances: A matrix of pairwise distance between each two item embeddings

        :return: three numpy arrays, holding the anchor index, positive index and
            negative index of each triplet, respectively
        """
        assert len(distances) == len(labels)

        from time import perf_counter

        import torch

        # labels = torch.tensor(labels)
        # distances = torch.tensor(distances)
        # labels1 = labels.unsqueeze(1)
        # labels2 = labels.unsqueeze(0)
        # matches = (labels1 == labels2).byte()
        # diffs = matches ^ 1
        # matches.fill_diagonal_(0)
        # triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
        # idxes_anchor, idxes_pos, idxes_neg = torch.where(triplets)
        # return idxes_anchor, idxes_pos, idxes_neg


        t0 = perf_counter()
        labels1 = np.expand_dims(labels, 1)
        labels2 = np.expand_dims(labels, 0)
        t1 = perf_counter()
        matches = (labels1 == labels2).astype(int)
        diffs = matches ^ 1
        np.fill_diagonal(matches, 0)
        t2 = perf_counter()
        triplets = np.expand_dims(matches, 2) * np.expand_dims(diffs, 1)
        t2a = perf_counter()
        idxes_anchor, idxes_pos, idxes_neg = np.where(triplets)
        t3 = perf_counter()
        T = t3 - t0
        # print(f'%% T={T:.2f} Tr={(t2a-t2)/T:.2%} W={(t3-t2a)/T:.2%}')
        return idxes_anchor, idxes_pos, idxes_neg


class SiameseSessionMiner(BaseMiner[Tuple[int, int]]):
    def mine(
        self, labels: Sequence[Tuple[int, int]], distances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate tuples from input embeddings and labels.

        :param labels: labels of each embeddings, consist of session id and label.
          the labels are either -1 or 1.
        :param distances: A matrix of pairwise distance between each two item embeddings

        :return: three numpy arrays, first one holding integers of first element of
            pair, second of the second element of pair, and third one the label (0 or
            1) for the pair for each pair
        """
        assert len(distances) == len(labels)

        ind_one, ind_two, labels_ret = [], [], []

        labels_with_index = [item + (index,) for index, item in enumerate(labels)]
        sorted_labels_with_index = sorted(labels_with_index, key=lambda x: x[0])

        for _, group in groupby(sorted_labels_with_index, lambda x: x[0]):
            for left, right in combinations(group, 2):  # (session_id, label, ind)
                if left[1] != -1 or right[1] != -1:
                    ind_one.append(left[2])
                    ind_two.append(right[2])
                    labels_ret.append(0 if min(left[1], right[1]) == -1 else 1)

        return np.array(ind_one), np.array(ind_two), np.array(labels_ret)


class TripletSessionMiner(BaseMiner[Tuple[int, int]]):
    def mine(
        self, labels: Sequence[Tuple[int, int]], distances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate triplets from input embeddings and labels.

        :param labels: labels of each embeddings, consist of session id and label,
            the labels of anchor, positive, negative are 0, 1 and -1.
        :param distances: A matrix of pairwise distance between each two item embeddings

        :return: three numpy arrays, holding the anchor index, positive index and
            negative index of each triplet, respectively
        """
        assert len(distances) == len(labels)

        anchor_ind, pos_ind, neg_ind = [], [], []

        labels_with_index = [item + (index,) for index, item in enumerate(labels)]
        sorted_labels_with_index = sorted(labels_with_index, key=lambda x: x[0])

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

        return np.array(anchor_ind), np.array(pos_ind), np.array(neg_ind)
