import json
from typing import List, Union, Optional, Dict

import numpy as np

NumericType = Union[
    int, float, complex, np.number
]  #: The type of numerics including numpy data type


class ScalarSequence:
    def __init__(self, name: str):
        """Create a record for storing a list of scalar values e.g. losses/metrics

        :param name: the name of that record
        """

        self.name = name
        self._record = []

    def __iadd__(self, other: Union[List[NumericType], float, 'ScalarSequence']):
        if isinstance(other, list):
            self._record += other
        elif isinstance(other, ScalarSequence):
            self._record += other._record
        elif isinstance(other, np.ndarray) and np.squeeze(other).ndim == 1:
            self._record += [v for v in np.squeeze(other)]
        else:
            self._record.append(other)
        return self

    def __str__(self):
        if self._record:
            return f'{self.name}: {np.mean([float(loss) for loss in self._record]):.2f}'
        else:
            return f'{self.name} has no record'

    def floats(self) -> List[NumericType]:
        """Return all numbers as a list of Python native float """
        return [float(v) for v in self._record]

    def __bool__(self):
        return bool(self._record)


class Summary:
    def __init__(self, *records: ScalarSequence):
        """Create a collection of summaries. """
        self._records = [r for r in records if r]

    def __iadd__(self, other: 'Summary'):
        if isinstance(other, Summary):
            self._records += other._records
        return self

    def save(self, filepath: str):
        """Store all summary into a JSON file"""
        with open(filepath, 'w') as fp:
            json.dump(
                self.dict(),
                fp,
            )

    def dict(self) -> Dict[str, List[NumericType]]:
        """Return all summaries as a Dictionary, where key is the name and value is the record"""
        return {r.name: r.floats() for r in self._records}

    def plot(
        self,
        output: Optional[str] = None,
        max_plot_points: Optional[int] = None,
        **kwargs,
    ):
        """Plot all records in the summary into one plot.

        .. note::
            This function requires ``matplotlib`` to be installed.

        :param output: Optional path to store the visualization. If not given, show in UI
        :param max_plot_points: the maximum number of points to plot. When the actual number of plots is larger than
            given number, then a linspace sampling is conducted first to get the actual number of points for plotting.
        :param kwargs: extra kwargs pass to matplotlib.plot
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            1,
            len(self._records),
            figsize=(6 * len(self._records), 6),
            constrained_layout=True,
        )
        if not isinstance(axes, np.ndarray):
            # when only one record, axes is not a list, so wrap it
            axes = [axes]

        plt_kwargs = dict(alpha=0.8, linewidth=1)
        plt_kwargs.update(kwargs)

        for idx, record in enumerate(self._records):
            axes[idx].plot(
                *self._sample_points(record.floats(), max_len=max_plot_points),
                **plt_kwargs,
            )
            axes[idx].set_ylabel(record.name)
            axes[idx].set_xlabel('Steps')

        if output:
            plt.savefig(output, bbox_inches='tight', pad_inches=0.1)
        else:
            plt.show()

    @staticmethod
    def _sample_points(arr, max_len: int):
        if not max_len or max_len > len(arr):
            return list(range(0, len(arr))), arr
        else:
            idx = np.round(np.linspace(0, len(arr) - 1, max_len)).astype(int)
            return idx, [arr[j] for j in idx]
