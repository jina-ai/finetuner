from typing import List
from itertools import combinations

from .base import BaseMiner


class SiameseMiner(BaseMiner):
    def mine(self, embeddings: List['TensorType'], labels: List[int]):
        """Generate tuples from input embeddings and labels, cut by limit if set.

        :param embeddings: embeddings from model, should be a list of Tensor objects.
        :param labels: labels of each embeddings, embeddings with same label indicates same class.
        :return: a pair of embeddings and their labels as tuple.
        """
        for left, right in combinations(enumerate(labels), 2):
            if left[1] == right[1]:
                yield embeddings[left[0]], embeddings[right[0]], 1
            else:
                yield embeddings[left[0]], embeddings[right[0]], -1


class TripletMiner(BaseMiner):
    def mine(self, embeddings: List['TensorType'], labels: List[int]):
        """Generate triplets from input embeddings and labels, cut by limit if set.

        :param embeddings: embeddings from model, should be a list of Tensor objects.
        :param labels: labels of each embeddings, embeddings with same label indicates same class.
        :return: triplet of embeddings follows the order of anchor, positive and negative.
        """
        for left, middle, right in combinations(enumerate(labels), 3):
            # two items share the same label (label1, label1, label2) -> (anchor, pos, neg)
            if left[1] == middle[1] and left[1] != right[1]:
                yield embeddings[left[0]], embeddings[middle[0]], embeddings[right[0]]
                yield embeddings[middle[0]], embeddings[left[0]], embeddings[right[0]]
            elif left[1] == right[1] and left[1] != middle[1]:
                yield embeddings[left[0]], embeddings[right[0]], embeddings[middle[0]]
                yield embeddings[right[0]], embeddings[left[0]], embeddings[middle[0]]
            elif middle[1] == right[1] and middle[1] != left[1]:
                yield embeddings[middle[0]], embeddings[right[0]], embeddings[left[0]]
                yield embeddings[right[0]], embeddings[right[0]], embeddings[left[0]]
