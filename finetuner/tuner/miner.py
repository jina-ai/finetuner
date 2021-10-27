from itertools import combinations

from .base import BaseMiner


class SiameseMiner(BaseMiner):
    def mine(self):
        """Generate tuples/triplets from input embeddings and labels, cut by limit if set."""
        rv = []
        for left, right in combinations(enumerate(self.labels), 2):
            if left[1] == right[1]:
                label = 1
            else:
                label = -1
            rv.append((self.embeddings[left[0]], self.embeddings[right[0]], label))
        return rv[: self.limit]


class TripletMiner(BaseMiner):
    def mine(self):
        pass
