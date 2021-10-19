import json
from typing import List, Union, Optional, Dict

import numpy as np

NumericType = Union[
    int, float, complex, np.number
]  #: The type of numerics including numpy data type


class ScalarSummary:
    def __init__(self, name: str = '', data: Optional[List[NumericType]] = None):
        """Create a record for storing a list of scalar values e.g. losses/metrics

        :param name: the name of that record
        :param data: the data record to initialize from
        """

        self._name = name or ''
        self._record = data or []

    def __iadd__(self, other: Union[List[NumericType], float, 'ScalarSummary']):
        if isinstance(other, list):
            self._record += other
        elif isinstance(other, ScalarSummary):
            self._record += other._record
        else:
            self._record.append(other)
        return self

    def __str__(self):
        if self._record:
            return (
                f'{self._name}: {np.mean([float(loss) for loss in self._record]):.2f}'
            )
        else:
            return f'{self._name} has no record'

    def floats(self) -> List[NumericType]:
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

    def dict(self) -> Dict[str, List[NumericType]]:
        """Return all summaries as a Dictionary, where key is the name and value is the record"""
        return {r._name: r.floats() for r in self._records}
