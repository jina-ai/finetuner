import numpy as np
from typing import List, Tuple
from itertools import combinations

from ..base import BaseMiner
from ...helper import AnyTensor


class SiameseMiner(BaseMiner):
    def mine(self, embeddings: List[AnyTensor], labels: List[int]) -> Tuple[int]:
        """Generate tuples from input embeddings and labels.

        :param embeddings: embeddings from model, should be a list of Tensor objects.
        :param labels: labels of each embeddings, embeddings with same label indicates same class.
        :return: a pair of label indices and their label as tuple.
        """
        assert len(embeddings) == len(labels)
        for left, right in combinations(enumerate(labels), 2):
            if left[1] == right[1]:
                yield left[0], right[0], 1
            else:
                yield left[0], right[0], -1


class TripletMiner(BaseMiner):
    def mine(self, embeddings: List[AnyTensor], labels: List[int]) -> Tuple[int]:
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
        for idx_anchor, idx_pos, idx_neg in zip(idxes_anchor, idxes_pos, idxes_neg):
            yield idx_anchor, idx_pos, idx_neg
