from itertools import combinations

from .base import BaseMiner


class SiameseMiner(BaseMiner):
    def mine(self):
        """Generate tuples/triplets from input embeddings and labels, cut by limit if set."""
        for left, right in combinations(enumerate(self.labels), 2)[: self.limit]:
            if left[1] == right[1]:
                yield self.embeddings[left[0]], self.embeddings[right[0]], 1
            else:
                yield self.embeddings[left[0]], self.embeddings[right[0]], -1


class TripletMiner(BaseMiner):
    def mine(self):
        pass
