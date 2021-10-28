import torch
import numpy as np
from typing import List, Generator
from itertools import combinations

from ..base import BaseMiner


class SiameseMiner(BaseMiner):
    def mine(self, embeddings: List[torch.Tensor], labels: List[int]):
        """Generate tuples from input embeddings and labels, cut by limit if set.

        :param embeddings: embeddings from model, should be a list of Tensor objects.
        :param labels: labels of each embeddings, embeddings with same label indicates same class.
        :return: a pair of embeddings and their labels as tuple.
        """
        assert len(embeddings) == len(labels)
        for left, right in combinations(enumerate(labels), 2):
            if left[1] == right[1]:
                yield embeddings[left[0]], embeddings[right[0]], 1
            else:
                yield embeddings[left[0]], embeddings[right[0]], -1


class TripletMiner(BaseMiner):
    def mine(self, embeddings: List[torch.Tensor], labels: List[int]):
        """Generate triplets from input embeddings and labels, cut by limit if set.

        :param embeddings: embeddings from model, should be a list of Tensor objects.
        :param labels: labels of each embeddings, embeddings with same label indicates same class.
        :return: triplet of embeddings follows the order of anchor, positive and negative.
        """
        assert len(embeddings) == len(labels)
        labels1 = np.expand_dims(labels, 1)
        labels2 = np.expand_dims(labels, 0)
        matches = (labels1 == labels2).astype(int)
        diffs = matches ^ 1
        np.fill_diagonal(matches, 0)
        triplets = np.expand_dims(matches, 2) * np.expand_dims(diffs, 1)
        indices_left, indices_middle, indices_right = np.where(triplets)
        for idx_left, idx_middle, idx_right in zip(
            indices_left, indices_middle, indices_right
        ):
            yield embeddings[idx_left], embeddings[idx_middle], embeddings[idx_right]
