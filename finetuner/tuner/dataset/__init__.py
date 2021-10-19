import itertools
from ... import __default_tag_key__
import numpy as np


class SiameseMixin:
    def __iter__(self):
        for d in self._inputs:
            d_blob = d.blob
            for m in d.matches:
                yield (d_blob, m.blob), np.float32(m.tags[__default_tag_key__]['label'])


class TripletMixin:
    def __iter__(self):
        for d in self._inputs:
            anchor = d.blob
            positives = []
            negatives = []
            for m in d.matches:
                if m.tags[__default_tag_key__]['label'] > 0:
                    positives.append(m.blob)
                else:
                    negatives.append(m.blob)

            for p, n in itertools.product(positives, negatives):
                yield (anchor, p, n), np.float32(0)
