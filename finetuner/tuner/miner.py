from itertools import combinations

from .base import BaseMiner


class SiameseMiner(BaseMiner):
    def mine(self, embeddings, labels):
        """Generate tuples from input embeddings and labels, cut by limit if set."""
        for left, right in combinations(enumerate(labels), 2):
            if left[1] == right[1]:
                yield embeddings[left[0]], embeddings[right[0]], 1
            else:
                yield embeddings[left[0]], embeddings[right[0]], -1


class TripletMiner(BaseMiner):
    def mine(self):
        """Generate triplets from input embeddings and labels, cut by limit if set."""
        for left, middle, right in combinations(enumerate(self.labels), 3):
            # two items share the same label (label1, label1, label2) -> (anchor, pos, neg)
            pass
