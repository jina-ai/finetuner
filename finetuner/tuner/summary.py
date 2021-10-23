import json
from collections import defaultdict
from typing import List, Union, Dict

import numpy as np

NumericType = Union[
    int, float, complex, np.number
]  #: The type of numerics including numpy data type


class ScalarSummary:
    def __init__(self, name: str = ''):
        """Create a record for storing a :py:class:`defaultdict` with lists.
        The key is the epoch number, the values are scalar values w.r.t the epoch, e.g. losses/metrics.

        :param name: the name of that record
        """

        self.name = name or ''
        self.record = defaultdict(list)

    def __iadd__(self, other: Union[Dict, 'ScalarSummary']):
        if isinstance(other, ScalarSummary):
            other = other.record
        for key, val in other.items():
            if isinstance(val, NumericType.__args__):
                self.record[key].append(float(val))
            elif isinstance(val, list):
                self.record[key].extend([float(item) for item in val])
            else:
                raise TypeError(f'Unexpected value type {type(val)}.')
        return self

    def _values(self):
        vals = []
        for item in self.record.values():
            vals += item
        return vals

    def __str__(self):
        if self.record:
            return f'{self.name}: {np.mean(self._values()):.2f}'
        else:
            return f'{self.name} has no record'


class SummaryCollection:
    def __init__(self, *records: ScalarSummary):
        """Create a collection of summaries."""
        self._records = records

    def save(self, filepath: str):
        """Store all summary into a JSON file"""
        with open(filepath, 'w') as fp:
            json.dump(
                self.dict(),
                fp,
            )

    def dict(self) -> Dict[str, List[NumericType]]:
        """Return all summaries as a Dictionary, where key is the name and value is the record"""
        return {r.name: dict(r.record) for r in self._records}

    def plot(self):
        """Draw learning curve."""
        pass
