import json
from typing import List, Union, Optional, Dict

import numpy as np


class ScalarSummary:
    def __init__(self, name: str, data: Optional[List[float]] = None):
        """Create a record for storing losses/metrics

        :param name: the name of that record
        """

        self._name = name
        self._record = data or []

    def __iadd__(self, other: Union[List[float], float, 'ScalarSummary']):
        if isinstance(other, list):
            self._record += other
        elif isinstance(other, ScalarSummary):
            self._record += other._record
        else:
            self._record.append(other)
        return self

    def __str__(self):
        return f'{self._name}: {np.mean([float(loss) for loss in self._record]):.2f}'

    def floats(self) -> List[float]:
        """Return all numbers as a list of Python native float """
        return [float(v) for v in self._record]


class SummaryCollection:
    def __init__(self, *records: ScalarSummary):
        """Create a collection of summaries. """
        self._records = records

    def save(self, filepath: str):
        """Store all summary into a JSON file"""
        with open(filepath, 'w') as fp:
            json.dump(
                self.dict(),
                fp,
            )

    def dict(self) -> Dict[str, List[float]]:
        """Return all summaries as a Dictionary, where key is the name and value is the record"""
        return {r._name: r.floats() for r in self._records}
